import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. RWKV-7 CORE COMPONENTS ---

@torch.jit.script
def rwkv7_recurrence(r, k, v, a, b, g, w, state):
    """
    RWKV-7 Generalized Delta Rule recurrence.
    Processes a sequence of inputs and updates the hidden state.
    """
    out_list = []
    for t in range(r.size(1)):
        # state = state * w + (k * v) + (a * b)
        state = state * w + (k[:, t, :] * v[:, t, :]) + (a[:, t, :] * b[:, t, :])
        out_list.append(state) 
    return torch.stack(out_list, dim=1), state

class GRN(nn.Module):
    """
    Global Response Normalization (from ConvNeXt v2).
    Encourages feature competition and diversity across channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        # x: [B, T, D]
        gx = torch.norm(x, p=2, dim=1, keepdim=True) # [B, 1, D]
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x

class RWKV7_Block(nn.Module):
    """
    Standard RWKV-7 Attention/Mixing Block.
    Includes Time-Mix (Generalized Delta Rule) and Channel-Mix (FFN + GRN).
    """
    def __init__(self, layer_id, dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.state_norm = nn.GroupNorm(dim // 32, dim)
        self.grn = GRN(dim)
        self.dropout = nn.Dropout(dropout)

        # Time-Mix
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.delta_a = nn.Linear(dim, dim, bias=False)
        self.delta_b = nn.Linear(dim, dim, bias=False)
        self.decay = nn.Parameter(torch.ones(dim) * -3.0)
        
        # Output projections
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.ffn_out_proj = nn.Linear(dim, dim, bias=False)

        # Channel-Mix (Simplified v7 Gating)
        self.ffn_key = nn.Linear(dim, dim * 4)
        self.ffn_value = nn.Linear(dim * 4, dim)
        self.ffn_receptance = nn.Linear(dim, dim)
        
        # Init
        for p in [self.key, self.value, self.delta_a, self.delta_b, self.out_proj, self.ffn_out_proj]:
            nn.init.normal_(p.weight, std=0.02)
        nn.init.orthogonal_(self.receptance.weight, gain=2.0)
        nn.init.orthogonal_(self.gate.weight, gain=2.0)

    def forward(self, x, state=None):
        xx = self.ln1(x)
        
        shifted_x = self.time_shift(xx)
        xm = xx * 0.5 + shifted_x * 0.5
        
        r = self.receptance(xm)
        k = self.key(xm)
        v = self.value(xm)
        a = self.delta_a(xm)
        b = self.delta_b(xm)
        g = torch.sigmoid(self.gate(xm))
        w = torch.exp(-torch.exp(self.decay))

        if state is None: 
            state = torch.zeros_like(xx[:, 0, :])
        
        states, state = rwkv7_recurrence(r, k, v, a, b, g, w, state)
        
        # Normalization and gating
        b_sz, t, d = states.size()
        norm_states = self.state_norm(states.reshape(b_sz * t, d)).reshape(b_sz, t, d)
        mixed_out = self.out_proj(torch.sigmoid(r) * norm_states * g)
        
        x = x + self.dropout(mixed_out)
        
        # MLP / Channel-Mix
        xx = self.ln2(x)
        k_ffn = torch.square(torch.relu(self.ffn_key(xx)))
        ffn_hidden = self.grn(self.ffn_value(k_ffn))
        ffn_out = self.ffn_out_proj(torch.sigmoid(self.ffn_receptance(xx)) * ffn_hidden)
        x = x + self.dropout(ffn_out)
        
        return x, state

# --- 2. TRANSDUCER COMPONENTS ---

class Conv2dSubsampling(nn.Module):
    """
    Subsamples the input spectrogram by a factor of 4x in time and frequency.
    Standard front-end for ASR models.
    """
    def __init__(self, in_channels, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.sub_h = 20 # Assuming 80 Mel bins -> 40 -> 20
        self.out_dim = 32 * self.sub_h
        self.linear = nn.Linear(self.out_dim, dim)

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.conv(x) 
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f) # (B, T/4, D')
        return self.linear(x)

class RWKV_Encoder(nn.Module):
    """
    RWKV-based Acoustic Encoder.
    Processes audio features (Mel spectrograms) into hidden representations.
    """
    def __init__(self, dim, n_layers, dropout=0.1):
        super().__init__()
        self.subsampling = Conv2dSubsampling(1, dim)
        self.ln_in = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([RWKV7_Block(i, dim, dropout=dropout) for i in range(n_layers)])

    def forward(self, x):
        # x: (B, F, T)
        x = x.unsqueeze(1) # (B, 1, F, T)
        x = self.subsampling(x)
        x = self.ln_in(x)
        for block in self.blocks:
            x, _ = block(x)
        return x

class RWKV_Predictor(nn.Module):
    """
    RWKV-based Label Predictor (LM component of Transducer).
    Processes previously predicted tokens.
    """
    def __init__(self, vocab_size, dim, n_layers, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.ln_in = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([RWKV7_Block(i, dim, dropout=dropout) for i in range(n_layers)])

    def forward(self, y, state=None):
        """
        y: [B, T] token IDs
        state: Optional list of states per layer
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
        """
        Stateful greedy decoding for RNN-T.
        Runs in O(U) by carrying the RNN state forward.
        """
        batch_size = enc_out.size(0)
        predictions = []

        for b in range(batch_size):
            y = torch.full((1, 1), blank_idx, dtype=torch.long, device=enc_out.device)
            state = None
            b_enc = enc_out[b:b+1]
            hyp = []
            
            # Initial predictor step for BOS
            pred_out, state = self.forward(y, state)
            
            t, u = 0, 0
            max_t = b_enc.size(1)
            max_u_per_t = 5
            max_hyp_len = 500
            
            while t < max_t and len(hyp) < max_hyp_len:
                logits = joiner(b_enc[:, t:t+1, :], pred_out[:, -1:, :])
                token = torch.argmax(logits, dim=-1).item()
                
                if token == blank_idx or u >= max_u_per_t:
                    t += 1
                    u = 0
                else:
                    hyp.append(token)
                    u += 1
                    y = torch.tensor([[token]], device=enc_out.device)
                    pred_out, state = self.forward(y, state)
            
            predictions.append(hyp)
        return predictions

class Transducer_Joiner(nn.Module):
    """
    Combines Encoder and Predictor outputs to predict the next token.
    Uses a Deep Joiner architecture with non-linear bottlenecks.
    """
    def __init__(self, dim, vocab_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim * 2, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Neutral bias for blank token
        with torch.no_grad():
            self.fc3.bias[vocab_size - 1] = 0.0

    def forward(self, enc_out, pred_out):
        t, u = enc_out.size(1), pred_out.size(1)
        enc_out = enc_out.unsqueeze(2).expand(-1, -1, u, -1)
        pred_out = pred_out.unsqueeze(1).expand(-1, t, -1, -1)
        
        combined = torch.cat([enc_out, pred_out], dim=-1)
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class RWKV_Transducer(nn.Module):
    """
    Full RWKV-T ASR Model.
    Supports RNN-T training (Lattice) and CTC auxiliary supervision.
    """
    def __init__(self, vocab_size=2049, dim=256, n_enc=12, n_pred=4, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.blank_idx = vocab_size - 1
        self.dim = dim
        self.encoder = RWKV_Encoder(dim, n_enc, dropout=dropout)
        self.predictor = RWKV_Predictor(vocab_size, dim, n_pred, dropout=dropout)
        self.joiner = Transducer_Joiner(dim, vocab_size, dropout=dropout)
        # CTC head for auxiliary training
        self.ctc_head = nn.Linear(dim, vocab_size)
        
        with torch.no_grad():
            self.ctc_head.bias.fill_(0.0)
            self.ctc_head.bias[self.blank_idx] = 1.0

    def forward(self, audio, labels):
        """Standard Forward for RNN-T training."""
        enc_out = self.encoder(audio)
        batch_size = labels.size(0)
        # Prepend blank (BOS)
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
        logits = self.forward_ctc(audio) 
        if blank_suppression > 0:
            logits[:, :, self.blank_idx] -= blank_suppression
        probs = F.softmax(logits, dim=-1)
        best_path = torch.argmax(probs, dim=-1)
        
        predictions = []
        for b in range(best_path.size(0)):
            hyp, prev = [], None
            t_limit = input_lengths[b] if input_lengths is not None else best_path.size(1)
            for token in best_path[b, :t_limit]:
                token = token.item()
                if token != self.blank_idx and token != prev:
                    hyp.append(token)
                prev = token
            predictions.append(hyp)
        return predictions
