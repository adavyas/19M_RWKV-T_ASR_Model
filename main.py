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
        self.grn = GRN(dim)
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
        self.decay = nn.Parameter(torch.ones(dim) * -3.0) # Slower decay for longer context
        
        # Output projections (Missing from earlier version)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.ffn_out_proj = nn.Linear(dim, dim, bias=False)

        # Channel-Mix (Simplified v7 Gating)
        # Hidden dim = 4 * dim = 1024
        self.ffn_key = nn.Linear(dim, 1024)
        self.ffn_value = nn.Linear(1024, dim)
        self.ffn_receptance = nn.Linear(dim, dim)
        
        # Initialization: Small std for state weights (std approx 0.01)
        for p in [self.key, self.value, self.delta_a, self.delta_b, self.out_proj, self.ffn_out_proj]:
            nn.init.normal_(p.weight, std=0.02)
        
        nn.init.orthogonal_(self.receptance.weight, gain=2.0)
        nn.init.orthogonal_(self.gate.weight, gain=2.0)

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
        
        # Apply normalization and gating
        b, t, d = states.size()
        norm_states = self.state_norm(states.view(b * t, d)).view(b, t, d)
        mixed_out = self.out_proj(torch.sigmoid(r) * norm_states * g)
        
        x = x + self.dropout(mixed_out)
        
        # MLP (Simplified v7 gating) with GRN
        xx = self.ln2(x)
        k_ffn = torch.square(torch.relu(self.ffn_key(xx)))
        ffn_hidden = self.grn(self.ffn_value(k_ffn))
        ffn_out = self.ffn_out_proj(torch.sigmoid(self.ffn_receptance(xx)) * ffn_hidden)
        x = x + self.dropout(ffn_out)
        
        return x, state

# --- 2. TRANSDUCER COMPONENTS ---
import torch
import torch.nn as nn

class Conv2dSubsampling(nn.Module):
    def __init__(self, in_channels, dim):
        super().__init__()
        # out_channels for the convs. 32 is standard for ASR front-ends.
        self.conv = nn.Sequential(
            # Layer 1: Downsample Freq (80->40) and Time (T->T/2)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Layer 2: Downsample Freq (40->20) and Time (T/2->T/4)
            # Stride 2 achieves the intended 4x temporal subsampling.
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.sub_h = 20 
        self.out_dim = 32 * self.sub_h
        
        self.linear = nn.Linear(self.out_dim, dim)

    def forward(self, x):
        # x input: (B, 1, 80, T)
        x = self.conv(x) # Output: (B, 32, 20, T/4)
        
        b, c, f, t = x.size()
        # Reshape for sequence processing: (B, T/4, Channels * Freq)
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        
        return self.linear(x)
class RWKV_Encoder(nn.Module):
    def __init__(self, dim, n_layers, dropout=0.1):
        super().__init__()
        # 4x temporal subsampling
        self.subsampling = Conv2dSubsampling(1, dim)
        self.ln_in = nn.LayerNorm(dim)
        
        # Proper blocks init
        self.blocks = nn.ModuleList([RWKV7_Block(i, dim, dropout=dropout) for i in range(n_layers)])

    def forward(self, x):
        # input x: (B, 80, T) from MelSpectrogram
        x = x.unsqueeze(1) # (B, 1, 80, T)
        x = self.subsampling(x)
        # Verify shape: (B, T', D)
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

    def forward(self, y, state=None):
        """Standard forward (lattice) or single-step (recurrent) inference.
        y: [B, T] token IDs
        state: Optional list of states per layer [L, B, D]
        """
        x = self.embed(y)
        x = self.ln_in(x)
        
        new_states = []
        for i, block in enumerate(self.blocks):
            s = state[i] if state is not None else None
            x, s = block(x, s)
            new_states.append(s)
            
        return x, new_states

    @torch.no_grad()
    def greedy_decode(self, enc_out, joiner, blank_idx):
        """Stateful greedy decoding for RNN-T.
        Runs in O(U) instead of O(U^2) by carrying the RNN state forward.
        """
        batch_size = enc_out.size(0)
        predictions = []

        for b in range(batch_size):
            # Start with BOS (which is usually the Blank token in RNN-T)
            y = torch.full((1, 1), blank_idx, dtype=torch.long, device=enc_out.device)
            state = None
            b_enc = enc_out[b:b+1] # [1, T, D]
            hyp = []
            
            # Pre-compute initial predictor output for BOS
            pred_out, state = self.forward(y, state) # pred_out: [1, 1, D]
            
            t = 0
            max_t = b_enc.size(1)
            u = 0 # Count tokens per frame
            max_u_per_t = 5
            max_hyp_len = 500 # Global safety cap
            
            # DEBUG: Check first frame probabilities
            debug_first = (b == 0)
            
            while t < max_t and len(hyp) < max_hyp_len:
                # Join current frame t and current predictor state
                logits = joiner(b_enc[:, t:t+1, :], pred_out[:, -1:, :])
                
                if debug_first and t == 0:
                    probs = F.softmax(logits, dim=-1)
                    blank_prob = probs[0, 0, 0, blank_idx].item()
                    top5_probs, top5_ids = torch.topk(probs[0, 0, 0], 5)
                    print(f"DEBUG greedy t=0: blank_prob={blank_prob:.4f}, top5_ids={top5_ids.tolist()}, top5_probs={top5_probs.tolist()}")
                    debug_first = False
                
                token = torch.argmax(logits, dim=-1).item()
                
                if token == blank_idx or u >= max_u_per_t:
                    t += 1
                    u = 0
                else:
                    hyp.append(token)
                    u += 1
                    # For the next predictor step, we only need the single new token
                    y = torch.tensor([[token]], device=enc_out.device)
                    pred_out, state = self.forward(y, state)
            
            predictions.append(hyp)
        return predictions

class Transducer_Joiner(nn.Module):
    def __init__(self, dim, vocab_size, dropout=0.1):
        super().__init__()
        # Deep Joiner: 3 layers with non-linear bottleneck
        self.fc1 = nn.Linear(dim * 2, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize blank bias neutral
        with torch.no_grad():
            self.fc3.bias[vocab_size - 1] = 0.0

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
        self.vocab_size = vocab_size
        self.blank_idx = vocab_size - 1
        self.dim = dim
        self.encoder = RWKV_Encoder(dim, n_enc, dropout=dropout)
        # Predictor handles actual tokens; blank usually acts as BOS
        self.predictor = RWKV_Predictor(vocab_size, dim, n_pred, dropout=dropout)
        self.joiner = Transducer_Joiner(dim, vocab_size, dropout=dropout)
        # CTC head
        self.ctc_head = nn.Linear(dim, vocab_size)
        
        # Favor blank slightly to break the 'babbling' local minimum early
        with torch.no_grad():
            self.ctc_head.bias.fill_(0.0)
            self.ctc_head.bias[self.blank_idx] = 1.0

    def forward(self, audio, labels):
        enc_out = self.encoder(audio)
        
        # Prepend a blank token (BOS) for the predictor history
        batch_size = labels.size(0)
        bos_token = torch.full((batch_size, 1), self.blank_idx, dtype=labels.dtype, device=labels.device)
        bos_labels = torch.cat([bos_token, labels], dim=1)
        
        pred_out, _ = self.predictor(bos_labels)
        return self.joiner(enc_out, pred_out)

    def forward_ctc(self, audio):
        enc_out = self.encoder(audio)
        return self.ctc_head(enc_out)

    @torch.no_grad()
    def greedy_decode(self, audio):
        self.eval()
        enc_out = self.encoder(audio)
        return self.predictor.greedy_decode(enc_out, self.joiner, self.blank_idx)

    @torch.no_grad()
    def greedy_decode_ctc(self, audio, input_lengths=None, blank_suppression=0.0):
        self.eval()
        logits = self.forward_ctc(audio) # [B, T, V]
        
        if blank_suppression > 0:
            logits[:, :, self.blank_idx] -= blank_suppression
            
        probs = F.softmax(logits, dim=-1)
        best_path = torch.argmax(probs, dim=-1) # [B, T]
        
        predictions = []
        for b in range(best_path.size(0)):
            hyp = []
            prev = None
            t_limit = input_lengths[b] if input_lengths is not None else best_path.size(1)
            for token in best_path[b, :t_limit]:
                token = token.item()
                if token != self.blank_idx and token != prev:
                    hyp.append(token)
                prev = token
            predictions.append(hyp)
        return predictions

class GRN(nn.Module):
    """Global Response Normalization (from ConvNeXt v2)
    Encourages feature competition and diversity across channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        # x: [B, T, D]
        # 1. Global feature magnitude
        gx = torch.norm(x, p=2, dim=1, keepdim=True) # [B, 1, D]
        # 2. Divide by mean to get relative response
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        # 3. Scale and residual
        return self.gamma * (x * nx) + self.beta + x

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

    # Test greedy decoding
    preds = model.greedy_decode(audio)
    print(f"Greedy Decode Sample: {preds[0]}")

    # Compute and print stats
    print_final_stats(model, time_steps=25, label_steps=20) # 25 frames after 4x subsampling