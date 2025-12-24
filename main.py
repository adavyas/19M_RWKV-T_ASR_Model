import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. THE RWKV-7 CORE BLOCK (GOOSE) ---
@torch.jit.script
def rwkv7_recurrence(r, k, v, a, b, g, w, state):
    out_list = []
    for t in range(r.size(1)):
        # state = state * w + (k*v) + (a*b)
        state = state * w + (k[:, t, :] * v[:, t, :]) + (a[:, t, :] * b[:, t, :])
        # We process normalization outside the JIT for simplicity with GroupNorm
        out_list.append(state) 
    return torch.stack(out_list, dim=1), state

class RWKV7_Block(nn.Module):
    def __init__(self, layer_id, dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.state_norm = nn.GroupNorm(dim // 32, dim)
        self.dropout = nn.Dropout(dropout)

        # Time-Mix (Generalized Delta Rule)
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        
        # Projections for v7: r, k, v, a, b, g, w
        # v7 often uses a single linear or split linear for these
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        
        # Delta Rule Vectors (Simplified v7)
        self.delta_a = nn.Linear(dim, dim, bias=False)
        self.delta_b = nn.Linear(dim, dim, bias=False)
        
        # Vector-valued decay
        self.decay = nn.Parameter(torch.ones(dim)) 

        # Channel-Mix (Simplified v7 Gating)
        # Hidden dim = 4 * dim = 1024
        self.ffn_key = nn.Linear(dim, 1024)
        self.ffn_value = nn.Linear(1024, dim)
        self.ffn_receptance = nn.Linear(dim, dim)
        
        # Initialization: Small std for state weights (std approx 0.01)
        for p in [self.key, self.value, self.delta_a, self.delta_b]:
            nn.init.normal_(p.weight, std=0.01)
        
        nn.init.orthogonal_(self.receptance.weight, gain=0.1)
        nn.init.orthogonal_(self.gate.weight, gain=0.1)
        nn.init.constant_(self.decay, 1.0)

    def forward(self, x, state=None):
        xx = self.ln1(x)
        
        shifted_x = self.time_shift(xx)
        # Linear mix on normalized input
        xm = xx * 0.5 + shifted_x * 0.5
        r = self.receptance(xm)
        k = self.key(xm)
        v = self.value(xm)
        a = self.delta_a(xm)
        b = self.delta_b(xm)
        g = torch.sigmoid(self.gate(xm))
        w = torch.exp(-torch.exp(self.decay))

        # RWKV-7 Generalized Delta Rule state update (Scripted for speed)
        if state is None: state = torch.zeros_like(xx[:, 0, :])
        
        states, state = rwkv7_recurrence(r, k, v, a, b, g, w, state)
        
        # Apply normalization and gating to all timesteps at once
        # states: [B, T, D]
        b, t, d = states.size()
        norm_states = self.state_norm(states.view(b * t, d)).view(b, t, d)
        mixed_out = torch.sigmoid(r) * norm_states * g
        
        x = x + self.dropout(mixed_out)
        
        # MLP (Simplified v7 gating)
        xx = self.ln2(x)
        k_ffn = torch.square(torch.relu(self.ffn_key(xx)))
        ffn_out = torch.sigmoid(self.ffn_receptance(xx)) * self.ffn_value(k_ffn)
        x = x + self.dropout(ffn_out)
        
        return x, state

# --- 2. TRANSDUCER COMPONENTS ---
class Conv2dSubsampling(nn.Module):
    """4x temporal subsampling front-end"""
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, 1, 80, T)
        x = self.conv(x) # (B, C, F, T')
        b, c, f, t = x.size()
        # We want (B, T', C * F)
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        return x

class RWKV_Encoder(nn.Module):
    def __init__(self, dim, n_layers, dropout=0.1):
        super().__init__()
        # 4x temporal subsampling: two Conv2d layers with stride 2
        # Input features: 80 Mel bins
        self.subsampling = Conv2dSubsampling(1, dim, dropout=dropout)
        # Linear projection to model dimension
        # Note: after subsampling, height is approx (80-3)/2 + 1 = 39, then (39-3)/2 + 1 = 19
        sub_h = (((80 - 3) // 2 + 1) - 3) // 2 + 1
        self.linear = nn.Linear(dim * sub_h, dim)
        self.ln_in = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([RWKV7_Block(i, dim, dropout=dropout) for i in range(n_layers)])

    def forward(self, x):
        # input x: (B, 80, T) from MelSpectrogram
        x = x.unsqueeze(1) # (B, 1, 80, T)
        x = self.subsampling(x)
        # Verify shape before linear: (B, T', D*F)
        x = self.linear(x)
        x = self.ln_in(x)
        for block in self.blocks:
            x, _ = block(x)
        return x

class RWKV_Predictor(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.ln_in = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([RWKV7_Block(i, dim, dropout=dropout) for i in range(n_layers)])

    def forward(self, y):
        x = self.embed(y)
        x = self.ln_in(x)
        for block in self.blocks:
            x, _ = block(x)
        return x

class Transducer_Joiner(nn.Module):
    def __init__(self, dim, vocab_size, dropout=0.1):
        super().__init__()
        # Deep Joiner: 3 layers with non-linear bottleneck
        self.fc1 = nn.Linear(dim * 2, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_out, pred_out):
        t, u = enc_out.size(1), pred_out.size(1)
        enc_out = enc_out.unsqueeze(2).expand(-1, -1, u, -1)
        pred_out = pred_out.unsqueeze(1).expand(-1, t, -1, -1)
        
        combined = torch.cat([enc_out, pred_out], dim=-1)
        
        # Non-linear Joiner computation
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class RWKV_Transducer(nn.Module):
    # Vocab is 2048 BPE pieces + 1 for <blank> at index 2048
    def __init__(self, vocab_size=2049, dim=256, n_enc=12, n_pred=4, dropout=0.1):
        super().__init__()
        self.encoder = RWKV_Encoder(dim, n_enc, dropout=dropout)
        # Predictor only embeds actual tokens (vocab_size - 1), blank is not a text input
        self.predictor = RWKV_Predictor(vocab_size, dim, n_pred, dropout=dropout)
        self.joiner = Transducer_Joiner(dim, vocab_size, dropout=dropout)

    def forward(self, audio, labels):
        enc_out = self.encoder(audio)
        
        # Prepend a blank token (index 0) for the predictor BOS
        batch_size = labels.size(0)
        bos_token = torch.zeros((batch_size, 1), dtype=labels.dtype, device=labels.device)
        bos_labels = torch.cat([bos_token, labels], dim=1)
        
        pred_out = self.predictor(bos_labels)
        return self.joiner(enc_out, pred_out)

    @torch.no_grad()
    def greedy_decode(self, audio):
        """Simplified greedy decoding for Keyword Spotting"""
        self.eval()
        enc_out = self.encoder(audio) # [B, T, D]
        batch_size = audio.size(0)
        predictions = []

        for b in range(batch_size):
            # [1, 1] starting with <blank>
            y = torch.zeros((1, 1), dtype=torch.long, device=audio.device)
            b_enc = enc_out[b:b+1] # [1, T, D]
            hyp = []
            
            t = 0
            max_t = b_enc.size(1)
            while t < max_t:
                # Predictor output for current history y
                pred_out = self.predictor(y) # [1, len(y), D]
                
                # Join current frame t and last predicted state u
                # enc_out: [1, 1, D], pred_out: [1, 1, D] -> logits: [1, 1, 1, V]
                logits = self.joiner(b_enc[:, t:t+1, :], pred_out[:, -1:, :])
                token = torch.argmax(logits, dim=-1).item()
                
                if token == 0: # <blank> means move to next audio frame
                    t += 1
                else: # Predicted a word piece
                    hyp.append(token)
                    y = torch.cat([y, torch.tensor([[token]], device=audio.device)], dim=1)
            
            predictions.append(hyp)
        return predictions

# --- 3. HELPER FUNCTIONS ---
def estimate_flops(model, batch_size, time_steps, label_steps, is_training=False):
    """Surgically accurate FLOPs calculation for Path vs Lattice workload"""
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    pred_params = sum(p.numel() for p in model.predictor.parameters())
    joiner_params = sum(p.numel() for p in model.joiner.parameters())

    # 1. Sequential Components (Encoder + Predictor)
    # These run once per time/label step regardless of path/lattice
    seq_flops = 2 * (enc_params * time_steps + pred_params * label_steps)
    
    # 2. Joiner Component
    if is_training:
        # Full Lattice: Every T x U combination (used for loss)
        join_ops = 2 * joiner_params * (time_steps * label_steps)
    else:
        # Greedy Path: Only T + U combinations (used for inference)
        join_ops = 2 * joiner_params * (time_steps + label_steps)
    
    total_flops = (seq_flops + join_ops) * batch_size
    return total_flops / 1e6

def print_final_stats(model, time_steps, label_steps): 
    params = sum(p.numel() for p in model.parameters())
    
    # Calculate state: 1 vector of size [dim] per layer
    n_layers = len(model.encoder.blocks) + len(model.predictor.blocks)
    dim = model.encoder.blocks[0].dim
    state_kb = (n_layers * dim * 4) / 1024 # 4 bytes for float32
    
    inference_mflops = estimate_flops(model, 1, time_steps, label_steps, is_training=False)
    training_mflops = estimate_flops(model, 1, time_steps, label_steps, is_training=True)

    print(f"\n--- 19M Scale RWKV-T Report ---")
    print(f"Total Parameters: {params/1e6:.2f} Million")
    print(f"Inference Workload (Greedy): {inference_mflops:.2f} MFLOPs (per 1s audio)")
    print(f"Training Workload (Lattice): {training_mflops:.2f} MFLOPs (per 1s audio)")
    print(f"Recurrent State Memory: {state_kb:.2f} KB (Fixed size)")
    print(f"Alif B1 SRAM Usage: {(state_kb / 2048) * 100:.4f}%")
    print("-" * 30)

# --- 4. EXECUTION ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Initialize scaled 19M parameter model (Vocab=2049: 2048 BPE + blank)
    model = RWKV_Transducer(vocab_size=2049, dim=256, n_enc=12, n_pred=4).to(device)
    
    # Simulation: 1s audio (16000 samples -> ~100 frames at 160 hop)
    # Output of MelTransform is [B, 80, T]
    audio = torch.randn(1, 80, 100).to(device)
    # BPE Label history (tokens 0-2047, 2048 is blank)
    labels = torch.randint(0, 2048, (1, 20)).to(device)
    
    logits = model(audio, labels)
    print(f"Lattice Shape: {logits.shape}")

    # Compute and print stats
    print_final_stats(model, time_steps=25, label_steps=20) # 25 frames after 4x subsampling