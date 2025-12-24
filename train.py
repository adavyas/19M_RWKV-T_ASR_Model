import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
from torchaudio.datasets import LIBRISPEECH
from torchaudio.functional import rnnt_loss
import os
import sentencepiece as spm
import yaml
import json
import time
import subprocess
import random
import traceback
import argparse



# --- 1. ENVIRONMENT SETUP ---
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Executing on device: {DEVICE}")

# --- FIX: Force stable audio backend to avoid torchcodec bug on Lambda/B200 ---
try:
    # Handle different torchaudio versions for listing backends
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
                # min(deletion, insertion, substitution)
                d[i, j] = min(d[i-1, j], d[i, j-1], d[i-1, j-1]) + 1
    return d[len(ref_words), len(hyp_words)].item() / max(1, len(ref_words))

# --- 2. DATA PREPROCESSING ---
class LibriCollate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # LibriSpeech returns: (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)
        waveforms, transcripts = [], []
        audio_lengths, label_lengths = [], []

        for wav, _, transcript, *_ in batch:
            waveforms.append(wav.squeeze(0))
            # Encode transcript to BPE IDs
            # SentencePiece pieces are 1-based usually if index 0 is reserved
            ids = self.tokenizer.EncodeAsIds(transcript.lower())
            targets = torch.tensor(ids, dtype=torch.long)
            transcripts.append(targets)
            
            # Correct length for torchaudio.transforms.MelSpectrogram(center=True)
            # T = 1 + (L // hop_length) 
            # Note: This matches the default behavior of the mel_transform used in train()
            audio_lengths.append(1 + (wav.size(-1) // 160))
            label_lengths.append(len(ids))

        # Pad audio and targets
        waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True).unsqueeze(1)
        transcripts = torch.nn.utils.rnn.pad_sequence(transcripts, batch_first=True, padding_value=0)
        
        return waveforms, transcripts, torch.tensor(audio_lengths, dtype=torch.int32), torch.tensor(label_lengths, dtype=torch.int32)

# --- 3. TRAINING FUNCTION ---
def train(limit_override=None):
    try:
        # Load Configuration
        with open("config.yaml", "r") as f: 
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config.yaml: {e}")
        return

    # Explicit Type Casting to avoid comparison errors
    try:
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
        config['monitoring']['log_interval'] = int(config['monitoring']['log_interval'])
        config['monitoring']['validation_hook_interval'] = int(config['monitoring']['validation_hook_interval'])
    except KeyError as e:
        print(f"Missing configuration key: {e}")
        return
    except ValueError as e:
        print(f"Invalid value in configuration: {e}")
        return

    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['paths']['log_file']), exist_ok=True)
    
    # Export Manifest
    if config['monitoring']['export_manifest']:
        manifest = {
            "timestamp": time.ctime(),
            "git_hash": get_git_hash(),
            "config": config,
            "device": str(DEVICE)
        }
        with open(config['paths']['manifest_file'], "w") as f:
            json.dump(manifest, f, indent=4)

    tokenizer = get_tokenizer(config['paths']['tokenizer_model'])
    collate_fn = LibriCollate(tokenizer)
    vocab_size = config['model']['vocab_size']

    import csv
    log_file = open(config['paths']['log_file'], "w", newline='')
    log_writer = csv.writer(log_file)
    # Header: epoch, batch, loss, grad_norm, train_wer, train_cer, val_wer, val_cer, spi, rtfx
    log_writer.writerow(["epoch", "batch", "loss", "grad_norm", "train_wer", "train_cer", "val_wer", "val_cer", "spi", "rtfx"])

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_mels=80, hop_length=160).to(DEVICE)
    # Point 3: Spectrogram Normalization
    spec_norm = nn.InstanceNorm1d(80).to(DEVICE)

    # Initialize 19M RWKV-v7 Model (dim=256)
    # Vocab from tokenizer + 1 for <blank> at index vocab_size
    from main import RWKV_Transducer, print_final_stats
    model = RWKV_Transducer(
        vocab_size=vocab_size, 
        dim=config['model']['dim'], 
        n_enc=config['model']['n_enc'], 
        n_pred=config['model']['n_pred'],
        dropout=config['model']['dropout']
    ).to(DEVICE)
    
    # 12KB Recap: 12 layers * 256 dim * 4 bytes = 12.288 KB
    # Post-subsampling workload: 100 raw frames -> 25 model steps
    print_final_stats(model, time_steps=25, label_steps=20)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])
    # Point 2: Cosine LR Schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['epochs'])
    
    # Mixed Precision Training (FP16/BF16 for B200)
    use_amp = (DEVICE.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)
    
    print(f"Loading LibriSpeech-100... Batch Size: {config['train']['batch_size']}")
    train_set = LIBRISPEECH("./data", url="train-clean-100", download=True)
    limit = limit_override if limit_override is not None else config['train'].get('train_limit_samples')
    if limit:
        print(f"Limiting training to {limit} samples...")
        train_set = torch.utils.data.Subset(train_set, range(min(limit, len(train_set))))

    train_loader = DataLoader(
        train_set, 
        batch_size=config['train']['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4
    )

    val_set = LIBRISPEECH("./data", url="dev-clean", download=True)
    val_loader = DataLoader(
        val_set,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    print(f"Starting Training on {DEVICE} ({len(train_set)} samples), validating on {len(val_set)} samples...")

    BLANK_IDX = vocab_size - 1 # 2048
    total_steps = 0

    for epoch in range(config['train']['epochs']):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (waveforms, targets, audio_lengths, target_lengths) in enumerate(train_loader):
            step_start = time.time()
            waveforms, targets = waveforms.to(DEVICE), targets.to(DEVICE)
            
            # --- DATA AUGMENTATION ---
            # Random Time Shift, Gain, and White Noise
            # (Keeping existing logic but applying to whole batch)
            # shift = torch.randint(-1600, 1600, (1,)).item()
            # waveforms = torch.roll(waveforms, shifts=shift, dims=-1)
            
            # gain = 0.8 + torch.rand(1).item() * 0.4
            # waveforms = waveforms * gain
            
            # snr = 10.0 + torch.rand(waveforms.size(0), device=DEVICE) * 20.0
            # noise = torch.randn_like(waveforms)
            # waveforms: (B, 1, T), snr: (B, 1) matches leading dims
            # waveforms = torchaudio.functional.add_noise(waveforms, noise, snr.unsqueeze(1))
            noise = torch.randn_like(waveforms) * 0.01
            snr = torch.randint(10, 30, (waveforms.size(0),)).to(DEVICE)
            waveforms = torchaudio.functional.add_noise(waveforms, noise, snr.unsqueeze(1))

            # 1. Audio -> Mel Spectrogram [B, 80, T]
            mel = mel_transform(waveforms).squeeze(1)
            # Apply InstanceNorm
            mel = spec_norm(mel.to(DEVICE))
            
            # 2. Forward pass with AMP (Mixed Precision)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                logits = model(mel, targets)
            
            # 3. Compute RNN-T Loss
            # Precise calculation for 4x subsampling (Conv2d with K=3, S=2, P=0)
            audio_lengths, target_lengths = audio_lengths.to(DEVICE), target_lengths.to(DEVICE)
            
            # Layer 1: stride 2, kernel 3
            l1 = torch.floor((audio_lengths.float() - 3) / 2) + 1
            # Layer 2: stride 2, kernel 3
            input_lengths = (torch.floor((l1 - 3) / 2) + 1).to(torch.int32)
            
            # Reduce lengths by 1 for safety margin against edge cases
            input_lengths = torch.clamp(input_lengths - 1, min=1)
            
            # Validate input_lengths >= target_lengths
            for i in range(input_lengths.size(0)):
                if input_lengths[i] <= target_lengths[i]:
                    # print(f"Warning: Truncating target for sample {i}")
                    target_lengths[i] = torch.clamp(input_lengths[i] - 1, min=0)

            # Final clamp
            input_lengths = torch.clamp(input_lengths, max=logits.size(1))
            target_lengths = torch.clamp(target_lengths, max=logits.size(2) - 1)

            # DYNAMIC ALIGNMENT: Ensure the metadata (lengths) and tensor (logits) are in sync
            # This prevents "input length mismatch" by forcing the longest sample to match the tensor size
            if input_lengths.max() < logits.size(1):
                input_lengths[torch.argmax(input_lengths)] = logits.size(1)

            # Check for NaNs causing crashes
            if torch.isnan(logits).any():
                print("PANIC: NaNs in logits! Skipping batch.")
                continue

            # Safety Check: Ensure targets dimension matches logits (U = L + 1)
            # This is a strict requirement of rnnt_loss
            if targets.size(1) != logits.size(2) - 1:
                # This should theoretically not happen given our model code, but good to be safe
                if targets.size(1) < logits.size(2) - 1:
                    pad_len = (logits.size(2) - 1) - targets.size(1)
                    targets = torch.nn.functional.pad(targets, (0, pad_len))

            # Move to contiguous memory (Critical for CUDA/MPS kernels)
            # This must be the VERY last step before the loss function
            logits = logits.contiguous()
            targets = targets.to(torch.int32).contiguous()
            input_lengths = input_lengths.contiguous()
            target_lengths = target_lengths.contiguous()
            
            # Hybrid check: move to CPU only if using MPS
            use_cpu_loss = (DEVICE.type == "mps")
            
            loss = rnnt_loss(
                logits=logits.cpu() if use_cpu_loss else logits,
                targets=targets.cpu() if use_cpu_loss else targets,
                logit_lengths=input_lengths.cpu() if use_cpu_loss else input_lengths,
                target_lengths=target_lengths.cpu() if use_cpu_loss else target_lengths,
                blank=BLANK_IDX,
                reduction='mean'
            )
            
            # No .to(DEVICE) needed if inputs were already on device
            if use_cpu_loss: loss = loss.to(DEVICE)
            
            # Scale loss for gradient accumulation
            accum_steps = config['train']['accum_steps']
            loss = loss / accum_steps

            # 4. Backward with AMP Scaling (accumulate gradients)
            scaler.scale(loss).backward()
            
            # Step optimizer only every accum_steps batches
            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                # Point 9: Robust Gradient Clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['train']['grad_clipping'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                grad_norm = torch.tensor(0.0) # Placeholder for logging

            step_end = time.time()
            spi = step_end - step_start
            # RTFx = (Total Audio Duration) / (Processing Time)
            # audio_lengths are frames (160 hop), so total samples = sum(audio_lengths * 160)
            total_audio_sec = (audio_lengths.sum().item() * 160) / 16000
            rtfx = total_audio_sec / spi

            step_end = time.time()
            spi = step_end - step_start
            total_audio_sec = (audio_lengths.sum().item() * 160) / 16000
            rtfx = total_audio_sec / spi

            # --- TRAINING METRIC SIGNAL (Subset of batch to maintain speed) ---
            train_wer_sig, train_cer_sig = 0.0, 0.0
            if batch_idx % config['monitoring']['log_interval'] == 0:
                model.eval()
                with torch.no_grad():
                    # Decode only the first 4 samples of the batch for a speed-efficient signal
                    peek_num = min(4, waveforms.size(0))
                    preds_sig = model.greedy_decode(mel[:peek_num])
                    for i in range(peek_num):
                        ref = tokenizer.decode_ids(targets[i, :target_lengths[i]].tolist())
                        hyp = tokenizer.decode_ids(preds_sig[i])
                        train_wer_sig += calculate_wer(ref, hyp)
                        train_cer_sig += calculate_wer(" ".join(list(ref)), " ".join(list(hyp)))
                    train_wer_sig /= peek_num
                    train_cer_sig /= peek_num
                model.train()

            # Logging (placeholder 0.0 for val metrics during training rows)
            log_writer.writerow([epoch, batch_idx, f"{loss.item():.6f}", f"{grad_norm.item():.6f}", 
                               f"{train_wer_sig:.4f}", f"{train_cer_sig:.4f}", "0.0", "0.0", f"{spi:.4f}", f"{rtfx:.2f}"])
            log_file.flush()

            if batch_idx % config['monitoring']['log_interval'] == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item()*config['train']['accum_steps']:.4f} | T-WER: {train_wer_sig:.2%} | SPI: {spi:.2f}s | RTFx: {rtfx:.1f}")

            # --- LINGUISTIC PEEK HOOK ---
            total_steps += 1
            if total_steps % config['monitoring']['validation_hook_interval'] == 0:
                print("\n --- Linguistic Peek (Validation Samples) ---")
                model.eval()
                with torch.no_grad():
                    # Pick 5 samples from dev-clean
                    val_subset = LIBRISPEECH("./data", url="dev-clean", download=True)
                    peek_indices = random.sample(range(len(val_subset)), config['monitoring']['num_validation_samples'])
                    for idx in peek_indices:
                        wav, sr, text, *_ = val_subset[idx]
                        mel_p = mel_transform(wav.to(DEVICE)).squeeze(1)
                        mel_p = spec_norm(mel_p)
                        pred_ids = model.greedy_decode(mel_p.unsqueeze(0))
                        pred_text = tokenizer.decode_ids(pred_ids[0])
                        print(f"GT: {text.lower()}")
                        print(f"PR: {pred_text}")
                        print("-" * 30)
                model.train()
                print("\n")
        
        avg_loss = epoch_loss / len(train_loader)
        
        # --- VALIDATION PASS ---
        model.eval()
        val_epoch_loss = 0
        total_wer = 0
        total_cer = 0
        correct_sequences = 0
        total_sequences = 0
        print(f"Running Validation...")
        with torch.no_grad():
            for val_batch_idx, (waveforms, targets, audio_lengths, target_lengths) in enumerate(val_loader):
                waveforms, targets = waveforms.to(DEVICE), targets.to(DEVICE)
                mel = mel_transform(waveforms).squeeze(1)
                mel = spec_norm(mel.to(DEVICE))
                
                # Model Inference
                with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                    logits = model(mel, targets)
                
                # Loss Calculation
                # Precise calculation for 4x subsampling
                l1_v = torch.floor((audio_lengths.float() - 3) / 2) + 1
                input_lengths = (torch.floor((l1_v - 3) / 2) + 1).to(torch.int32)
                
                # Safety margins
                input_lengths = torch.clamp(input_lengths - 1, min=1)
                for i in range(input_lengths.size(0)):
                    if input_lengths[i] <= target_lengths[i]:
                        target_lengths[i] = torch.clamp(input_lengths[i] - 1, min=0)

                input_lengths = torch.clamp(input_lengths, max=logits.size(1))
                target_lengths = torch.clamp(target_lengths, max=logits.size(2) - 1)
                
                # Dynamic Alignment
                if input_lengths.max() < logits.size(1):
                    input_lengths[torch.argmax(input_lengths)] = logits.size(1)

                loss = rnnt_loss(
                    logits=logits.cpu() if use_cpu_loss else logits,
                    targets=targets.to(torch.int32).cpu() if use_cpu_loss else targets.to(torch.int32),
                    logit_lengths=input_lengths.cpu() if use_cpu_loss else input_lengths,
                    target_lengths=target_lengths.cpu() if use_cpu_loss else target_lengths,
                    blank=BLANK_IDX,
                    reduction='mean'
                )
                val_epoch_loss += loss.item()

                # Metrics Calculation
                preds = model.greedy_decode(mel)
                for i, p in enumerate(preds):
                    # Decode PR and GT
                    actual_target = targets[i, :target_lengths[i]].tolist()
                    ref_text = tokenizer.decode_ids(actual_target)
                    hyp_text = tokenizer.decode_ids(p)
                    
                    # Compute WER/CER
                    total_wer += calculate_wer(ref_text, hyp_text)
                    total_cer += calculate_wer(" ".join(list(ref_text)), " ".join(list(hyp_text)))
                    
                    if p == actual_target:
                        correct_sequences += 1
                    total_sequences += 1

        avg_val_loss = val_epoch_loss / len(val_loader)
        avg_wer = total_wer / total_sequences
        avg_cer = total_cer / total_sequences
        val_acc = correct_sequences / total_sequences

        print(f"--- Epoch {epoch} | Val Loss: {avg_val_loss:.4f} | WER: {avg_wer:.2%} | CER: {avg_cer:.2%} | LR: {scheduler.get_last_lr()[0]:.2e} ---")
        
        # Log summary row for the epoch
        log_writer.writerow([epoch, "VAL_SUMMARY", f"{avg_val_loss:.6f}", "0.0", 
                           "0.0", "0.0", f"{avg_wer:.4f}", f"{avg_cer:.4f}", "0.0", "0.0"])
        log_file.flush()
        
        scheduler.step()
        torch.save(model.state_dict(), f"{config['paths']['checkpoint_dir']}/rwkv_epoch_{epoch}.pth")
    
    log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RWKV-T Training Script")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of training samples (for testing)")
    args = parser.parse_args()

    try:
        train(limit_override=args.limit)
    except Exception:
        print("\n--- Training Interrupted by Fatal Error ---")
        traceback.print_exc()
