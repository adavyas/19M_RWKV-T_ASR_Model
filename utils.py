import torch
import sentencepiece as spm
import subprocess
import os

def get_tokenizer(model_path):
    """Loads the SentencePiece tokenizer model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Tokenizer model {model_path} not found.")
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp

def get_git_hash():
    """Returns the current git hash of the repository."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return "no-git-repo"

def calculate_wer(reference, hypothesis):
    """
    Computes Word Error Rate (WER) using Levenshtein distance.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    
    # DP table
    d = torch.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    for i in range(len(ref_words) + 1): d[i, 0] = i
    for j in range(len(hyp_words) + 1): d[0, j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]: 
                d[i, j] = d[i-1, j-1]
            else: 
                d[i, j] = min(d[i-1, j], d[i, j-1], d[i-1, j-1]) + 1
                
    return d[len(ref_words), len(hyp_words)].item() / len(ref_words)

def estimate_flops(model, batch_size, time_steps, label_steps, is_training=False):
    """
    Surgically accurate FLOPs calculation for RWKV-T.
    """
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    pred_params = sum(p.numel() for p in model.predictor.parameters())
    joiner_params = sum(p.numel() for p in model.joiner.parameters())

    # Sequential Components (Encoder + Predictor)
    seq_flops = 2 * (enc_params * time_steps + pred_params * label_steps)
    
    # Joiner Component
    if is_training:
        # Full Lattice: Every T x U combination
        join_ops = 2 * joiner_params * (time_steps * label_steps)
    else:
        # Greedy Path: Only T + U combinations
        join_ops = 2 * joiner_params * (time_steps + label_steps)
    
    total_flops = (seq_flops + join_ops) * batch_size
    return total_flops / 1e6

def print_final_stats(model, time_steps, label_steps): 
    """Prints a summary report of model parameters and workload."""
    params = sum(p.numel() for p in model.parameters())
    
    # State memory calculation
    n_layers = len(model.encoder.blocks) + len(model.predictor.blocks)
    dim = model.encoder.blocks[0].dim
    state_kb = (n_layers * dim * 4) / 1024 
    
    inference_mflops = estimate_flops(model, 1, time_steps, label_steps, is_training=False)
    training_mflops = estimate_flops(model, 1, time_steps, label_steps, is_training=True)

    print(f"\n--- RWKV-T Model Report ---")
    print(f"Total Parameters: {params/1e6:.2f} Million")
    print(f"Inference Workload (Greedy): {inference_mflops:.2f} MFLOPs (per 1s audio)")
    print(f"Training Workload (Lattice): {training_mflops:.2f} MFLOPs (per 1s audio)")
    print(f"Recurrent State Memory: {state_kb:.2f} KB (Fixed)")
    print("-" * 30)
