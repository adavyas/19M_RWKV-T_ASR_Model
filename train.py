import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import math
import torchaudio
from torchaudio.datasets import LIBRISPEECH
# Try k2 for pruned RNN-T, fall back to torchaudio rnnt_loss
try:
    import k2
    USE_K2 = True
    USE_TORCHAUDIO_RNNT = False
    print("Using k2 for pruned RNN-T loss")
except ImportError:
    USE_K2 = False
    from torchaudio.functional import rnnt_loss as torchaudio_rnnt_loss
    USE_TORCHAUDIO_RNNT = True
    print("k2 not available, using torchaudio.rnnt_loss on CPU")
import os
import sentencepiece as spm
import yaml
import json
import time
import subprocess
import random
import traceback
import argparse
import concurrent.futures
import csv
import wandb
from main import RWKV_Transducer, print_final_stats
from dataloader import get_dataloaders

# --- 1. ENVIRONMENT SETUP ---
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1" # Provides better tracebacks for memory errors
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Executing on device: {DEVICE}")

# Hardware Discovery for Mac/MPS
if DEVICE.type == "mps":
    import platform
    print(f"Hardware: Apple {platform.processor()} ({platform.machine()})")
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"System RAM: {mem.total / (1024**3):.1f}GB total / {mem.available / (1024**3):.1f}GB available")
    except ImportError:
        pass

# --- FIX: Force stable audio backend to avoid torchcodec bug on Lambda/B200 ---
try:
    print(f"Torchaudio version: {torchaudio.__version__}")
    if hasattr(torchaudio, "list_audio_backends"):
        backends = torchaudio.list_audio_backends()
    elif hasattr(torchaudio, "utils") and hasattr(torchaudio.utils, "list_audio_backends"):
        backends = torchaudio.utils.list_audio_backends()
    else:
        backends = []
    
    if "sox_io" in backends:
        if hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend("sox_io")
    elif "soundfile" in backends:
        if hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend("soundfile")
            
    if hasattr(torchaudio, "get_audio_backend"):
        print(f"Using audio backend: {torchaudio.get_audio_backend()}")
    else:
        print("Audio backend set (v2.1+ migration in progress).")
except Exception as e:
    print(f"Warning: Could not explicitly set audio backend: {e}")

def get_tokenizer(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Tokenizer model {model_path} not found. Run tokenizer_trainer.py first.")
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp

def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return "no-git-repo"

def calculate_wer(reference, hypothesis):
    """Simple WER implementation using Levenshtein distance"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    d = torch.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    for i in range(len(ref_words) + 1): d[i, 0] = i
    for j in range(len(hyp_words) + 1): d[0, j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]: 
                d[i, j] = d[i-1, j-1]
            else: 
                d[i, j] = min(d[i-1, j], d[i, j-1], d[i-1, j-1]) + 1
    return d[len(ref_words), len(hyp_words)].item() / max(1, len(ref_words))

# --- 2. DATA PREPROCESSING MOVED TO DATALOADER.PY ---

# --- 3. TRAINING FUNCTION ---
def train(limit_override=None, mode="joint", batch_size_override=None, fresh=False):
    try:
        with open("config.yaml", "r") as f: 
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config.yaml: {e}")
        return

    # Casting
    config['train']['lr'] = float(config['train']['lr'])
    config['train']['epochs'] = int(config['train']['epochs'])
    config['train']['batch_size'] = int(config['train']['batch_size'])
    config['train']['accum_steps'] = int(config['train'].get('accum_steps', 1))
    config['train']['weight_decay'] = float(config['train'].get('weight_decay', 0.1))
    config['train']['grad_clipping'] = float(config['train'].get('grad_clipping', 1.0))
    config['model']['dim'] = int(config['model']['dim'])
    config['model']['n_enc'] = int(config['model']['n_enc'])
    config['model']['n_pred'] = int(config['model']['n_pred'])
    config['model']['vocab_size'] = int(config['model']['vocab_size'])

    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['paths']['log_file']), exist_ok=True)
    
    if config['monitoring']['export_manifest']:
        manifest = {"timestamp": time.ctime(), "git_hash": get_git_hash(), "config": config, "device": str(DEVICE)}
        with open(config['paths']['manifest_file'], "w") as f: json.dump(manifest, f, indent=4)

    tokenizer = get_tokenizer(config['paths']['tokenizer_model'])
    vocab_size = config['model']['vocab_size']

    # --- WANDB INITIALIZATION ---
    if config['monitoring'].get('wandb_project'):
        try:
            wandb.init(
                project=config['monitoring']['wandb_project'],
                name=f"rwkv-t-19m-{time.strftime('%Y%m%d-%H%M%S')}",
                config=config
            )
        except Exception as e:
            print(f"Warning: WandB initialization failed: {e}")

    
    log_file = open(config['paths']['log_file'], "w", newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "batch", "loss", "grad_norm", "train_wer", "train_cer", "val_wer", "val_cer", "spi", "rtfx"])

    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, hop_length=160).to(DEVICE)
    spec_norm = nn.InstanceNorm1d(80).to(DEVICE)

    
    model = RWKV_Transducer(
        dim=config['model']['dim'],
        n_enc=config['model']['n_enc'],
        n_pred=config['model']['n_pred'],
        vocab_size=config['model']['vocab_size'],
        dropout=config['model'].get('dropout', 0.1)
    ).to(DEVICE)
    
    # --- MODULAR TRAINING: Freeze/Unfreeze components based on mode ---
    if mode == "encoder":
        print("Mode: Encoder Pre-training (CTC only)")
        # Freeze Predictor and Joiner
        for p in model.predictor.parameters(): p.requires_grad = False
        for p in model.joiner.parameters(): p.requires_grad = False
        # Ensure Encoder and CTC Head are unfrozen
        for p in model.encoder.parameters(): p.requires_grad = True
        for p in model.ctc_head.parameters(): p.requires_grad = True
    elif mode == "predictor":
        print("Mode: Predictor Pre-training (LM only)")
        # Freeze Encoder and CTC Head
        for p in model.encoder.parameters(): p.requires_grad = False
        for p in model.ctc_head.parameters(): p.requires_grad = False
        # Ensure Predictor and Joiner (for output layer) are unfrozen
        for p in model.predictor.parameters(): p.requires_grad = True
        for p in model.joiner.parameters(): p.requires_grad = True
    else:
        print("Mode: Joint Training (RNN-T + Auxiliary CTC)")
        # Everything unfrozen
        for p in model.parameters(): p.requires_grad = True
    
    # --- WARM-START STRATEGY ---
    rwkv_params = []
    new_params = []
    for name, param in model.named_parameters():
        if "blocks" in name:
            rwkv_params.append(param)
            param.requires_grad = False # Freeze initially
        else:
            new_params.append(param)
            param.requires_grad = True

    print(f"Warm-start: Freezing {len(rwkv_params)} RWKV block parameters for 1000 samples.")
    print(f"Warm-start: Training {len(new_params)} new/head parameters at LR 2e-4.")

    optimizer = optim.AdamW([
        {'params': rwkv_params, 'lr': 5e-4}, 
        {'params': new_params, 'lr': 2e-4}
    ], weight_decay=config['train']['weight_decay'], foreach=True)
    use_amp = (DEVICE.type != 'cpu')

    # --- COMPATIBLE GRADSCALER ---
    if DEVICE.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    elif hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(enabled=use_amp)
    else:
        scaler = None # No scaler available/needed
    
    # --- EPOCH-BASED LR SCHEDULER ---
    # ExponentialLR reduces LR by 10% after each epoch to help "hone in" on the minimum.
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    # --- LR WARMUP SCHEDULER ---
    warmup_steps = 200  # Longer warmup for higher LR
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    
    # --- TORCH COMPILE (Disabled for warp-rnnt testing) ---
    # if hasattr(torch, "compile") and DEVICE.type == 'cuda':
    #     print("Enabling torch.compile() for RWKV kernels...", flush=True)
    #     try:
    #         model = torch.compile(model)
    #     except Exception as e:
    #         print(f"torch.compile failed, falling back to eager: {e}")

    from main import estimate_flops
    print_final_stats(model, time_steps=25, label_steps=20)
    # Estimate Training GFLOPS/sec (approximate)
    train_mflops = estimate_flops(model, batch_size=config['train']['batch_size'], time_steps=25, label_steps=20, is_training=True)
    print(f"Estimated Training Workload: {train_mflops / 1000:.2f} GFLOPs per batch of 1s audio")
    
    start_epoch = 0
    # Checkpoint Safety: Mode-specific defaults
    default_save = "latest.pth"
    if mode == "encoder": default_save = "ctc_pretrain.pth"
    elif mode == "predictor": default_save = "predictor_lm.pth"
    
    BLANK_IDX = vocab_size - 1 
    total_steps = 0
    total_samples = 0
    unfrozen = False

    save_name = args.output if hasattr(args, 'output') and args.output != "latest.pth" else default_save
    checkpoint_path = os.path.join(config['paths']['checkpoint_dir'], save_name if os.path.exists(os.path.join(config['paths']['checkpoint_dir'], save_name)) else "latest.pth")
    
    if os.path.exists(checkpoint_path) and not fresh:
        print(f"Found checkpoint at {checkpoint_path}. Loading with strict=False...", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # Filter out mismatched keys (e.g. front-end/linear layers changed for 4x)
        state_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        filtered_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                filtered_dict[k] = v
            else:
                print(f"Skipping key {k} due to mismatch or missing")
        
        # Load with strict=False
        missing, unexpected = model.load_state_dict(filtered_dict, strict=False)
        print(f"Checkpoint Load: Successfully loaded {len(filtered_dict)} keys.")
        print(f"Checkpoint Load: Missing keys (will be fresh): {len(missing)}")
        # We don't load optimizer state here to avoid state mismatch with new layers
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint: start_epoch = checkpoint['epoch'] + 1
        if 'step' in checkpoint: total_steps = checkpoint['step']
        if 'samples' in checkpoint: total_samples = checkpoint['samples']
        if 'unfrozen' in checkpoint: unfrozen = checkpoint['unfrozen']
        
        if unfrozen or not fresh:
             print(f"Resuming in {'UNFROZEN' if unfrozen else 'WARM-START'} state ({'Backbone active' if unfrozen else f'{1000-total_samples} samples remaining'}).")
             for p in rwkv_params: p.requires_grad = True
             unfrozen = True
             
        if 'scaler_state_dict' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    total_tokens = 0 # Initialize token counter
    grad_norm = torch.tensor(0.0) # Move grad_norm here to persist last value
    num_workers = config['train'].get('num_workers', 4) 
    batch_size = batch_size_override if batch_size_override is not None else config['train']['batch_size']
    max_duration = config['train'].get('max_audio_duration', 15.0)
    limit = limit_override if limit_override is not None else config['train'].get('train_limit_samples')

    train_loader, val_loader, train_set, val_set = get_dataloaders(
        mode=mode,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        limit_samples=limit,
        max_audio_duration=max_duration
    )

    num_epochs = args.epochs if hasattr(args, 'epochs') and args.epochs is not None else config['train']['epochs']

    # Create a persistent iterator for streaming datasets to avoid redundant data every epoch
    train_iter = iter(train_loader) if isinstance(train_set, IterableDataset) else None

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        
        # Determine source: persistent iterator or fresh loader
        loader_source = range(1000) if train_iter is not None else train_loader
        
        for batch_idx, batch in enumerate(loader_source):
            if train_iter is not None:
                waveforms, targets, audio_lengths, target_lengths, langs = next(train_iter)
            else:
                waveforms, targets, audio_lengths, target_lengths, langs = batch
                
            step_start = time.time()
            total_samples += waveforms.size(0)
            
            if mode != "predictor":
                waveforms = waveforms.to(DEVICE, non_blocking=True)
                audio_lengths = audio_lengths.to(DEVICE, non_blocking=True)
                # Simple Augment: White Noise
                noise = torch.randn_like(waveforms) * 0.01
                snr = torch.randint(10, 30, (waveforms.size(0),)).to(DEVICE)
                waveforms = torchaudio.functional.add_noise(waveforms, noise, snr.unsqueeze(1))
            
            targets = targets.to(DEVICE)
            target_lengths = target_lengths.to(DEVICE)

            dtype = torch.bfloat16 if (DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
            
            # --- COMMON PRE-CALCULATIONS ---
            target_lengths_clamped = target_lengths
            mel = None 
            input_lengths = None

            # --- FORWARD PASS & LOSS ---
            with torch.amp.autocast(device_type=DEVICE.type, dtype=dtype, enabled=use_amp):
                # 1. Feature Extraction (Encoder/Joint only)
                if mode != "predictor":
                    mel = mel_transform(waveforms)[:, 0, :, :]
                    mel = torch.log1p(mel)
                    mel = spec_norm(mel)
                    if model.training and random.random() < 0.5:
                        # Simple SpecAugment
                        f_mask_size = random.randint(0, 15)
                        f_mask_pos = random.randint(0, 80 - f_mask_size)
                        mel[:, f_mask_pos:f_mask_pos+f_mask_size, :] = 0.0
                        
                        t_max = mel.size(2)
                        t_mask_size = random.randint(0, min(40, t_max // 4))
                        t_mask_pos = random.randint(0, t_max - t_mask_size)
                        mel[:, :, t_mask_pos:t_mask_pos+t_mask_size] = 0.0

                # 2. Loss Calculation by Mode
                if mode == "encoder":
                    logits = model.forward_ctc(mel)
                    input_lengths = torch.ceil((audio_lengths.float() / audio_lengths.max().float()) * logits.size(1)).to(torch.int32)
                    input_lengths = torch.clamp(input_lengths, min=1, max=logits.size(1))
                    
                    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
                    # Force CPU for CTC on Mac to avoid MPS NotImplementedError
                    if DEVICE.type == 'mps':
                        loss = F.ctc_loss(log_probs.float().cpu(), targets.cpu(), input_lengths.cpu(), target_lengths_clamped.cpu(), blank=BLANK_IDX).to(DEVICE)
                    else:
                        loss = F.ctc_loss(log_probs.float(), targets, input_lengths, target_lengths_clamped, blank=BLANK_IDX)
                
                elif mode == "predictor":
                    # For Predictor LM training: 
                    # Inputs: [SOS, t1, t2, ..., tn]
                    # Targets: [t1, t2, ..., tn, EOS/PAD]
                    # We want to predict t1 from SOS, t2 from t1, etc.
                    sos = torch.full((targets.size(0), 1), BLANK_IDX, device=DEVICE, dtype=targets.dtype)
                    inputs_full = torch.cat([sos, targets], dim=1) # [B, L+1]
                    
                    # Forward pass
                    pred_out, _ = model.predictor(inputs_full) # [B, L+1, D]
                    fake_enc = torch.zeros((pred_out.size(0), 1, model.dim), device=DEVICE, dtype=pred_out.dtype)
                    logits_full = model.joiner(fake_enc, pred_out).squeeze(1) # [B, L+1, V]
                    
                    # Align according to teacher forcing:
                    # predictions[i] predicts targets[i+1]
                    # So predictions = logits_full[:, :-1] 
                    # And labels = inputs_full[:, 1:] (which is just our original 'targets')
                    # This ensures logits_full[0] (from SOS) predicts inputs_full[1] (t1)
                    logits = logits_full[:, :-1, :]
                    labels = targets # [B, L]
                    
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=0)
                    
                    # --- Predictor Metrics ---
                    train_ppl = math.exp(loss.item()) if loss.item() < 20 else 1e9
                    with torch.no_grad():
                        preds = torch.argmax(logits, dim=-1)
                        mask = (labels != 0)
                        correct = (preds == labels) & mask
                        train_acc = correct.sum().item() / mask.sum().item() if mask.sum() > 0 else 0.0
                
                else: # Joint Mode
                    # RNN-T Loss (CPU Fallback)
                    logits = model(mel, targets)
                    input_lengths = torch.ceil((audio_lengths.float() / audio_lengths.max().float()) * logits.size(1)).to(torch.int32)
                    input_lengths = torch.clamp(input_lengths, min=1, max=logits.size(1))
                    
                    log_probs = F.log_softmax(logits.float(), dim=-1).cpu()
                    loss = torchaudio_rnnt_loss(log_probs, targets.cpu(), input_lengths.cpu(), target_lengths_clamped.cpu(), blank=BLANK_IDX)
                    
                    # Auxiliary CTC
                    ctc_logits = model.forward_ctc(mel)
                    ctc_log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
                    # Force CPU for CTC on Mac
                    if DEVICE.type == 'mps':
                        ctc_loss = F.ctc_loss(ctc_log_probs.float().cpu(), targets.cpu(), input_lengths.cpu(), target_lengths_clamped.cpu(), blank=BLANK_IDX).to(DEVICE)
                    else:
                        ctc_loss = F.ctc_loss(ctc_log_probs.float(), targets, input_lengths, target_lengths_clamped, blank=BLANK_IDX)
                    
                    loss = loss.to(DEVICE) + 0.7 * ctc_loss
            
            accum_steps = config['train']['accum_steps']
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf loss at batch {batch_idx}. Skipping backward pass.")
                optimizer.zero_grad()
                continue
                
            scaler.scale(loss / accum_steps).backward()
            
            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['grad_clipping'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Warmup scheduler step (first 200 optimizer steps)
                if total_steps < warmup_steps:
                    warmup_scheduler.step()

            epoch_loss += loss.item()
            total_steps += 1
            
            # --- TOKEN TRACKING ---
            batch_tokens = target_lengths.sum().item()
            total_tokens += batch_tokens

            # --- VIRTUAL EPOCH END ---
            if isinstance(train_set, IterableDataset) and batch_idx >= 999:
                print(f"\n[Virtual Epoch] 1000 batches reached. Rotating to next validation/checkpoint phase...", flush=True)

            # --- DYNAMIC UNFREEZING ---
            if total_samples >= 1000 and not unfrozen:
                print(f"\n>>> Warm-start phase complete ({total_samples} samples). Unfreezing RWKV blocks...", flush=True)
                for p in rwkv_params:
                    p.requires_grad = True
                unfrozen = True

            # --- PERIODIC SAVE ---
            if total_steps % 500 == 0:
                save_name = args.output if hasattr(args, 'output') and args.output != "latest.pth" else default_save
                ckpt_path = os.path.join(config['paths']['checkpoint_dir'], save_name)
                torch.save({
                    'epoch': epoch, 
                    'step': total_steps,
                    'samples': total_samples,
                    'unfrozen': unfrozen,
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'scaler_state_dict': scaler.state_dict() if scaler else None
                }, ckpt_path)
                print(f"Periodic checkpoint saved at step {total_steps}", flush=True)

            # --- REPAIRED METRICS BLOCK ---
            train_wer_sig, train_cer_sig = 0.0, 0.0
            spi, rtfx = 0.0, 0.0
            
            if batch_idx % config['monitoring']['log_interval'] == 0:
                spi = time.time() - step_start
                if mode == "predictor":
                    # For LM mode, RTFx is calculated as if the transcript was spoken
                    # LibriSpeech is ~150 words/min. 4x subsampling = 25 FPS = 0.04s per frame.
                    # We estimate 'audio duration' as tokens * 0.1s (rough average) to give RTFx a real meaning.
                    total_audio_sec = target_lengths.sum().item() * 0.1 
                else:
                    total_audio_sec = (audio_lengths.sum().item()) / 16000
                
                rtfx = total_audio_sec / max(1e-6, spi)
                
                # Dynamic Memory Tracking (MPS specific)
                mem_str = ""
                mem_mb = 0.0
                if DEVICE.type == "mps":
                    # MPS doesn't have a direct 'vram' tool like nvidia-smi, but psutil gives process memory
                    try:
                        import psutil
                        process = psutil.Process(os.getpid())
                        mem_mb = process.memory_info().rss / (1024**2)
                        mem_str = f" | Mem: {mem_mb:.1f}MB"
                    except: pass
                
                model.eval()
                peek_num = min(2, waveforms.size(0))
                with torch.no_grad():
                    if mode != "predictor":
                        # Autocast needed here too for the encoder pass
                        with torch.amp.autocast(device_type=DEVICE.type, dtype=dtype, enabled=use_amp):
                            # Peek logic: Use clean mel (no SpecAugment) for an accurate reading
                            clean_mel = mel_transform(waveforms[:peek_num])[:, 0, :, :]
                            clean_mel = torch.log1p(clean_mel)
                            clean_mel = spec_norm(clean_mel)
                            
                            # Debug: check the actual probability of the blank token and top tokens
                            logits = model.forward_ctc(clean_mel[:1])
                            probs = torch.softmax(logits[0, 0], dim=-1)
                            blank_prob = probs[BLANK_IDX].item()
                            top5_probs, top5_ids = torch.topk(probs, 5)
                            
                            if batch_idx % (config['monitoring']['log_interval'] * 10) == 0:
                                top5_pieces = [tokenizer.id_to_piece(tid.item()) if tid.item() < len(tokenizer) else "<blank>" for tid in top5_ids]
                                print(f"DEBUG greedy t=0: blank_prob={blank_prob:.4f}, top5_ids={top5_ids.tolist()}, top5_pieces={top5_pieces}, top5_probs={top5_probs.tolist()}")
                            
                            # Truthful Peek: No blank suppression
                            preds = model.greedy_decode_ctc(clean_mel, input_lengths=input_lengths[:peek_num], blank_suppression=0.0)
                        
                        for i in range(peek_num):
                            ref = tokenizer.decode_ids(targets[i, :target_lengths[i]].tolist())
                            hyp = tokenizer.decode_ids(preds[i])
                            train_wer_sig += calculate_wer(ref, hyp)
                            train_cer_sig += calculate_wer(" ".join(list(ref)), " ".join(list(hyp)))
                            # Immediate feedback every 10 log intervals
                            if batch_idx % (config['monitoring']['log_interval'] * 10) == 0:
                                print(f"Batch {batch_idx} | [{langs[i]}] GT: {ref[:50]}... | PR: {hyp[:50]}...")
                                print(f"Batch {batch_idx} | CER: {calculate_wer(' '.join(list(ref)), ' '.join(list(hyp))):.1%}")
                        train_wer_sig /= peek_num
                        train_cer_sig /= peek_num
                    else:
                        # LM Peek: Show some target vs predicted text
                        with torch.amp.autocast(device_type=DEVICE.type, dtype=dtype, enabled=use_amp):
                            # Decode the argmax of logits to see what the model is thinking
                            # logits is [B, U, V]
                            pred_ids = torch.argmax(logits, dim=-1) # [B, U]
                            for i in range(peek_num):
                                t_len = int(target_lengths[i].item())
                                # Safety check for garbage lengths or indices
                                if t_len > 1000 or t_len <= 0: # Arbitrary max length to catch bad data
                                    print(f"LM [{langs[i]}] Ref {i}: <invalid length {t_len}>")
                                else:
                                    try:
                                        t_ids = targets[i, :t_len].tolist()
                                        # Filter IDs to valid range 0 to vocab_size-1
                                        t_ids = [tid if (0 <= tid < vocab_size) else 0 for tid in t_ids]
                                        ref = tokenizer.decode_ids(t_ids)
                                        print(f"LM [{langs[i]}] Ref {i}: {ref[:60]}...")
                                    except Exception as e:
                                        print(f"LM [{langs[i]}] Ref {i}: <decode error: {e}>")

                                # Clamp t_len to logits size to avoid Indexing error if padding differs
                                t_len = min(t_len, pred_ids.size(1))
                                hyp = tokenizer.decode_ids(pred_ids[i, :t_len].tolist())
                                if batch_idx % (config['monitoring']['log_interval'] * 10) == 0:
                                    print(f"LM [{langs[i]}] Prd {i}: {hyp[:60]}...")
                model.train()
                
                if mode == "predictor":
                    print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f} | PPL: {train_ppl:.2f} | Acc: {train_acc:.1%} | Grad: {grad_norm.item():.2f} | Tok: {batch_tokens} | SPI: {spi:.2f}s | RTFx: {rtfx:.1f} | Mem: {mem_mb:.1f}MB", flush=True)
                else:
                    print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f} | WER: {train_wer_sig:.1%} | CER: {train_cer_sig:.1%} | Grad: {grad_norm.item():.2f} | SPI: {spi:.2f}s | RTFx: {rtfx:.1f} | Mem: {mem_mb:.1f}MB", flush=True)

                if wandb.run:
                    log_data = {
                        "train/loss": loss.item(),
                        "train/grad_norm": grad_norm.item(),
                        "train/rtfx": rtfx,
                        "train/spi": spi,
                        "train/total_tokens": total_tokens,
                        "train/batch_tokens": batch_tokens,
                        "epoch": epoch
                    }
                    if mode == "predictor":
                        log_data.update({
                            "train/ppl": train_ppl,
                            "train/acc": train_acc
                        })
                    else:
                        log_data.update({
                            "train/wer": train_wer_sig,
                            "train/cer": train_cer_sig,
                        })
                    wandb.log(log_data)
                
                log_writer_row_data = [epoch, batch_idx, f"{loss.item():.6f}", f"{grad_norm.item():.6f}"]
                if mode == "predictor":
                    log_writer_row_data.extend([f"{train_ppl:.2f}", f"{train_acc:.4f}", "0.0", "0.0"]) # PPL, Acc, WER, CER
                else:
                    log_writer_row_data.extend([f"{train_wer_sig:.4f}", f"{train_cer_sig:.4f}", "0.0", "0.0"]) # WER, CER, PPL, Acc (PPL/Acc are 0 for non-predictor)
                log_writer_row_data.extend([f"{spi:.4f}", f"{rtfx:.2f}"]) # SPI, RTFx
                log_writer.writerow(log_writer_row_data)
                log_file.flush()

            if total_steps % config['monitoring']['validation_hook_interval'] == 0:
                model.eval()
                with torch.no_grad():
                    # 1. Select a validation sample
                    if isinstance(val_set, IterableDataset):
                        sample = next(iter(val_set))
                    else:
                        idx = random.randint(0, len(val_set)-1)
                        sample = val_set[idx]
                    
                    if mode == "predictor":
                        v_text, v_lang = sample
                        v_wav = None
                    else:
                        v_wav, _, v_text, *v_rest = sample if isinstance(sample, (tuple, list)) else (None, None, sample, "en")
                        v_lang = v_rest[-1] if v_rest else "en"
                    
                    if mode != "predictor":
                        v_mel = mel_transform(v_wav.unsqueeze(0).to(DEVICE))
                        v_mel = spec_norm(torch.log1p(v_mel[:, 0, :, :]))
                        v_hyp = tokenizer.decode_ids(model.greedy_decode_ctc(v_mel)[0] if mode == "encoder" else model.greedy_decode(v_mel)[0])
                    else:
                        v_ids = tokenizer.EncodeAsIds(v_text.lower())
                        # This part of the original code was incomplete for predictor mode validation peek
                        # It's hard to fix without knowing the exact model.predictor and model.joiner usage for greedy decoding.
                        # For now, I'll keep the original structure but acknowledge it might need more logic.
                        # A proper greedy decode for LM would involve feeding predicted tokens back.
                        # This is a simplified placeholder.
                        lm_input_peek = torch.full((1, 1), BLANK_IDX, device=DEVICE, dtype=torch.long)
                        pred_out_peek, _ = model.predictor(lm_input_peek)
                        fake_enc_peek = torch.zeros((1, 1, model.dim), device=DEVICE, dtype=pred_out_peek.dtype)
                        lm_logits_peek = model.joiner(fake_enc_peek, pred_out_peek).squeeze(1)
                        v_hyp = tokenizer.decode_ids(torch.argmax(lm_logits_peek, dim=-1)[0].tolist())
                    
                    print(f"\n--- {mode.upper()} PEEK ---")
                    print(f"GT: {v_text.lower()[:100]}...")
                    print(f"PR: {v_hyp[:100]}...")
                    print("-" * 30)
                model.train()
        
        model.eval()
        val_loss, val_wer, val_cer, val_cnt = 0, 0, 0, 0
        val_ppl = 0
        val_acc = 0 # Initialize val_acc
        with torch.no_grad():
            for val_batch_idx, (v_wavs, v_tgts, v_alen, v_tlen, v_langs) in enumerate(val_loader):
                if val_batch_idx >= 5: break
                v_tgts = v_tgts.to(DEVICE)
                v_tlen_clamped = v_tlen.to(torch.int32).to(DEVICE)
                
                if mode == "predictor":
                    # SOS Parity Fix: Prepend Blank as SOS to match inference
                    sos = torch.full((v_tgts.size(0), 1), BLANK_IDX, device=DEVICE, dtype=v_tgts.dtype)
                    lm_input = torch.cat([sos, v_tgts[:, :-1]], dim=1)
                    lm_targets = v_tgts # Define lm_targets for validation
                    
                    with torch.amp.autocast(device_type=DEVICE.type, dtype=dtype, enabled=use_amp):
                        pred_out, _ = model.predictor(lm_input)
                        fake_enc = torch.zeros((pred_out.size(0), 1, pred_out.size(-1)), device=DEVICE, dtype=pred_out.dtype)
                        lm_logits = model.joiner(fake_enc, pred_out).squeeze(1) # [B, U, V]
                        v_loss = F.cross_entropy(lm_logits.reshape(-1, lm_logits.size(-1)), lm_targets.reshape(-1), ignore_index=0)
                    val_loss += v_loss.item()
                    val_ppl += torch.exp(v_loss).item()
                    
                    # Calculate validation accuracy for predictor mode
                    preds = torch.argmax(lm_logits, dim=-1)
                    mask = (lm_targets != 0)
                    correct = (preds == lm_targets) & mask
                    val_acc += correct.sum().item() / mask.sum().item() if mask.sum() > 0 else 0.0
                    val_cnt += 1 # Increment val_cnt for each batch in predictor mode
                else:
                    v_wavs = v_wavs.to(DEVICE)
                    v_mel = mel_transform(v_wavs)[:, 0, :, :]
                    v_mel = torch.log1p(v_mel)
                    v_mel = spec_norm(v_mel)
                    
                    v_alen, v_tlen = v_alen.to(DEVICE), v_tlen.to(DEVICE)
                
                if mode == "encoder":
                    v_logits = model.forward_ctc(v_mel)
                    v_log_probs = F.log_softmax(v_logits, dim=-1).transpose(0, 1)

                    # Length calculation for CTC
                    v_max_t = v_logits.size(1)
                    v_batch_max_samples = v_alen.max().float()
                    v_ilen = torch.ceil((v_alen.float() / v_batch_max_samples) * v_max_t).to(torch.int32)
                    v_ilen = torch.clamp(v_ilen, min=1, max=v_max_t)
                    v_tlen_clamped = v_tlen.to(torch.int32)

                    if DEVICE.type == 'mps':
                        v_loss = F.ctc_loss(v_log_probs.float().cpu(), v_tgts.cpu(), v_ilen.cpu(), v_tlen_clamped.cpu(), blank=BLANK_IDX, reduction='mean').to(DEVICE)
                    else:
                        v_loss = F.ctc_loss(v_log_probs.float(), v_tgts, v_ilen, v_tlen_clamped, blank=BLANK_IDX, reduction='mean')
                    
                    val_loss += v_loss.item()
                    
                    # Truthful WER (no suppression) with proper input lengths
                    v_preds = model.greedy_decode_ctc(v_mel, input_lengths=v_ilen, blank_suppression=0.0)
                    for i in range(len(v_preds)):
                        ref = tokenizer.decode_ids(v_tgts[i, :v_tlen_clamped[i]].tolist())
                        hyp = tokenizer.decode_ids(v_preds[i])
                        val_wer += calculate_wer(ref, hyp)
                        val_cer += calculate_wer(" ".join(list(ref)), " ".join(list(hyp)))
                        val_cnt += 1
                elif mode == "predictor":
                    pass # Handled above
                else:
                    # Joint Validation
                    v_logits = model(v_mel, v_tgts)
                    
                    v_max_t = v_logits.size(1)
                    v_max_u = v_logits.size(2)
                    v_batch_max_samples = v_alen.max().float()
                    v_ilen = torch.ceil((v_alen.float() / v_batch_max_samples) * v_max_t).to(torch.int32)
                    v_ilen = torch.clamp(v_ilen, min=1, max=v_max_t)
                    v_tlen_clamped = torch.clamp(v_tlen, min=0, max=v_max_u - 1).to(torch.int32)
                    
                    v_rnnt_loss = torchaudio_rnnt_loss(
                        logits=v_logits.float().cpu(),
                        targets=v_tgts.to(torch.int32).cpu(),
                        logit_lengths=v_ilen.cpu(),
                        target_lengths=v_tlen_clamped.cpu(),
                        blank=BLANK_IDX, reduction='mean'
                    )
                    
                    v_ctc_logits = model.forward_ctc(v_mel)
                    v_ctc_log_probs = F.log_softmax(v_ctc_logits, dim=-1).transpose(0, 1)
                    
                    if DEVICE.type == 'mps':
                        v_ctc_loss = F.ctc_loss(
                            v_ctc_log_probs.float().cpu(), v_tgts.cpu(), 
                            v_ilen.cpu(), v_tlen_clamped.cpu(),
                            blank=BLANK_IDX, reduction='mean'
                        ).to(DEVICE)
                    else:
                        v_ctc_loss = F.ctc_loss(
                            v_ctc_log_probs.float(), v_tgts, v_ilen, v_tlen_clamped,
                            blank=BLANK_IDX, reduction='mean'
                        )
                    
                    val_loss += (v_rnnt_loss.to(DEVICE) + 0.7 * v_ctc_loss).item()
                    
                    # Joint Mode Specific Metric
                    v_preds = model.greedy_decode(v_mel)
                    for i, p in enumerate(v_preds):
                        val_wer += calculate_wer(tokenizer.decode_ids(v_tgts[i, :v_tlen[i]].tolist()), tokenizer.decode_ids(p))
                        val_cnt += 1
                
                # Metrics are now calculated inside the mode blocks
                pass
        
        if mode == "predictor":
            avg_loss = val_loss / max(1, val_cnt)
            avg_ppl = val_ppl / max(1, val_cnt)
            avg_acc = val_acc / max(1, val_cnt)
            print(f"\nFinal Validation | Loss: {avg_loss:.4f} | PPL: {avg_ppl:.2f} | Acc: {avg_acc:.1%}")
            if wandb.run:
                wandb.log({"val/loss": avg_loss, "val/ppl": avg_ppl, "val/acc": avg_acc, "epoch": epoch})
        else:
            avg_loss = val_loss / max(1, val_cnt)
            avg_wer = val_wer / max(1, val_cnt)
            avg_cer = val_cer / max(1, val_cnt)
            print(f"\nFinal Validation | Loss: {avg_loss:.4f} | WER: {avg_wer:.1f}% | CER: {avg_cer:.1f}%")
            if wandb.run:
                wandb.log({"val/loss": avg_loss, "val/wer": avg_wer, "val/cer": avg_cer, "epoch": epoch})
        scheduler.step()
        save_name = args.output if hasattr(args, 'output') and args.output else "latest.pth"
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict()}, f"{config['paths']['checkpoint_dir']}/{save_name}")
    
    log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mode", type=str, default="joint", choices=["encoder", "predictor", "joint"])
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train")
    parser.add_argument("--output", type=str, default="latest.pth", help="Output checkpoint filename")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    parser.add_argument("--fresh", action="store_true", help="Start training from scratch (ignore checkpoints)")
    args = parser.parse_args()
    
    train(limit_override=args.limit, mode=args.mode, batch_size_override=args.batch_size, fresh=args.fresh)
