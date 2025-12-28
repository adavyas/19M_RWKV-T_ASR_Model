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
import concurrent.futures
import csv
import wandb
from main import RWKV_Transducer, print_final_stats

# --- 1. ENVIRONMENT SETUP ---
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1" # Provides better tracebacks for memory errors
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Executing on device: {DEVICE}")

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

# --- 2. DATA PREPROCESSING ---
class LibriCollate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        waveforms, transcripts = [], []
        audio_lengths, label_lengths = [], []

        for wav, _, transcript, *_ in batch:
            waveforms.append(wav.squeeze(0))
            ids = self.tokenizer.EncodeAsIds(transcript.lower())
            targets = torch.tensor(ids, dtype=torch.long)
            transcripts.append(targets)
            
            # Correct length for 4x subsampling in main.py
            audio_lengths.append(wav.size(-1)) # Send raw samples, subsample logic handled in loss block
            label_lengths.append(len(ids))

        waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True).unsqueeze(1)
        transcripts = torch.nn.utils.rnn.pad_sequence(transcripts, batch_first=True, padding_value=0)
        
        return waveforms, transcripts, torch.tensor(audio_lengths, dtype=torch.int32), torch.tensor(label_lengths, dtype=torch.int32)

# --- 3. TRAINING FUNCTION ---
def train(limit_override=None):
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
    collate_fn = LibriCollate(tokenizer)
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
        vocab_size=vocab_size, 
        dim=config['model']['dim'], 
        n_enc=config['model']['n_enc'], 
        n_pred=config['model']['n_pred'],
        dropout=config['model'].get('dropout', 0.1)
    ).to(DEVICE)
    
    # --- OPTIMIZED COMPONENTS FOR B200 ---
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['train']['lr'], 
        weight_decay=config['train']['weight_decay'],
        foreach=True # Significant speedup on B200
    )
    use_amp = (DEVICE.type != 'cpu')

    # --- COMPATIBLE GRADSCALER ---
    if DEVICE.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    elif hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(enabled=use_amp)
    else:
        scaler = None # No scaler available/needed
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['epochs'])
    
    # --- TORCH COMPILE (Disabled temporarily to debug CUDA Illegal Access) ---
    # if hasattr(torch, "compile") and DEVICE.type == 'cuda':
    #     print("Enabling torch.compile() for RWKV kernels...", flush=True)
    #     try:
    #         model = torch.compile(model)
    #     except Exception as e:
    #         print(f"torch.compile failed: {e}")

    print_final_stats(model, time_steps=25, label_steps=20)
    
    start_epoch = 0
    checkpoint_path = os.path.join(config['paths']['checkpoint_dir'], "latest.pth")
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}. Loading...", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint: start_epoch = checkpoint['epoch'] + 1
        if 'scaler_state_dict' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    num_workers = config['train'].get('num_workers', 4) #B200 handls 12 by default; use 4 for A10 testing
    max_duration = config['train'].get('max_audio_duration', 15.0)
    max_samples = int(max_duration * 16000)

    print(f"Loading LibriSpeech-100 from {os.path.abspath('./data')}...", flush=True)
    train_set = LIBRISPEECH("./data", url="train-clean-100", download=True)
    
    raw_count = len(train_set)
    print(f"Dataset initialized. Found {raw_count} raw samples.", flush=True)
    
    if raw_count == 0:
        print("ERROR: Dataset is empty! please check if data/LibriSpeech/train-clean-100 exists and contains .flac files.")
        return

    # --- WORKER-POWERED FILTERING ---
    print(f"Filtering samples > {max_duration}s using {num_workers} workers...", flush=True)
    
    # --- SMART PATH DISCOVERY ---
    sample_rel = train_set._walker[0]
    s_id, c_id, _ = sample_rel.split("-")
    
    # Check common patterns
    found_root = None
    candidates = [
        os.path.join("./data", "LibriSpeech", "train-clean-100"),
        os.path.join("./data", "train-clean-100"),
        os.path.join("./data", "LibriSpeech"),
        "./data"
    ]
    
    for cand in candidates:
        test_path = os.path.join(cand, s_id, c_id, f"{sample_rel}.flac")
        if os.path.exists(test_path):
            found_root = cand
            print(f"Path Discovery: Found data at {found_root}", flush=True)
            break
            
    if not found_root:
        print("ERROR: Could not locate .flac files. Please check your data directory structure.")
        return

    first_error = None
    
    def check_length(idx):
        nonlocal first_error
        try:
            rel_path = train_set._walker[idx]
            speaker_id, chapter_id, _ = rel_path.split("-")
            full_path = os.path.join(found_root, speaker_id, chapter_id, f"{rel_path}.flac")
            
            if not os.path.exists(full_path):
                if first_error is None: first_error = f"File not found: {full_path}"
                return None
            
            # Try multiple ways to get info
            if hasattr(torchaudio, "info"):
                info = torchaudio.info(full_path)
            elif hasattr(torchaudio, "backend") and hasattr(torchaudio.backend, "common"):
                # Fallback for very old internal structures
                info = torchaudio.backend.common.AudioMetaData(full_path)
            else:
                # If we truly can't get info, just keep the file (don't filter)
                return idx
                
            return idx if info.num_frames < max_samples else None
        except Exception as e:
            if first_error is None: first_error = f"torchaudio.info failed on {rel_path}: {e}"
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(check_length, range(raw_count)))
    
    valid_indices = [r for r in results if r is not None]
    print(f"Filter complete: {len(valid_indices)}/{raw_count} samples remaining.", flush=True)
    
    if len(valid_indices) == 0:
        print(f"ERROR: All samples were filtered out!")
        print(f"DEBUG: First error encountered: {first_error}")
        print("-" * 30)
        print("POSSIBLE FIXES:")
        print("1. Run 'du -sh data/' to check if files are actually downloaded (~6.4GB).")
        print("2. Run 'sudo apt-get install sox libsox-fmt-all' if on Linux.")
        print("-" * 30)
        return

    train_set = torch.utils.data.Subset(train_set, valid_indices)

    limit = limit_override if limit_override is not None else config['train'].get('train_limit_samples')
    if limit: train_set = torch.utils.data.Subset(train_set, range(min(limit, len(train_set))))

    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_set = LIBRISPEECH("./data", url="dev-clean", download=True) # dev-clean acts as validation. train-clean-100 is used for training.
    val_loader = DataLoader(val_set, batch_size=config['train']['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    BLANK_IDX = vocab_size - 1 
    total_steps = 0

    for epoch in range(start_epoch, config['train']['epochs']):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (waveforms, targets, audio_lengths, target_lengths) in enumerate(train_loader):
            step_start = time.time()
            # non_blocking=True overlaps transfer with next batch loading
            waveforms, targets = waveforms.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            audio_lengths, target_lengths = audio_lengths.to(DEVICE, non_blocking=True), target_lengths.to(DEVICE, non_blocking=True)
            
            # Simple Augment: White Noise
            noise = torch.randn_like(waveforms) * 0.01
            snr = torch.randint(10, 30, (waveforms.size(0),)).to(DEVICE)
            waveforms = torchaudio.functional.add_noise(waveforms, noise, snr.unsqueeze(1))

            # --- FORCED B200 STABILITY OVERRIDES ---
            torch.cuda.synchronize() 
            torch.cuda.empty_cache()
            
            dtype = torch.bfloat16 if (DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
            with torch.amp.autocast(device_type=DEVICE.type, dtype=dtype, enabled=use_amp):
                mel = mel_transform(waveforms).squeeze(1)
                mel = spec_norm(mel)
                logits = model(mel, targets)
                
                # --- PHASE 2: LOGIT CAPPING (Prevent numerical overflow) ---
                logits = torch.clamp(logits, min=-10.0, max=10.0)
            
            # --- DIAGNOSTIC LOGGING (First Batch Only) ---
            if batch_idx == 0 and epoch == start_epoch:
                print(f"DEBUG: Logits shape: {logits.shape}, dtype: {logits.dtype}, device: {logits.device}")
                print(f"DEBUG: Targets shape: {targets.shape}, dtype: {targets.dtype}, device: {targets.device}")
                if torch.isnan(logits).any():
                    print("CRITICAL: Logits contain NaNs! Model is unstable.")
            
            # --- SAFE DYNAMIC LENGTH CALCULATION ---
            max_t = logits.size(1)
            max_u = logits.size(2)
            batch_max_samples = audio_lengths.max().float()
            input_lengths = torch.ceil((audio_lengths.float() / batch_max_samples) * max_t).to(torch.int32)
            input_lengths = torch.clamp(input_lengths, min=1, max=max_t)
            target_lengths = torch.clamp(target_lengths, min=0, max=max_u - 1)

            for b_idx in range(input_lengths.size(0)):
                if input_lengths[b_idx] <= target_lengths[b_idx]:
                    input_lengths[b_idx] = torch.clamp(target_lengths[b_idx] + 1, max=max_t)

            # --- TOTAL STABILITY LOSS CALL ---
            use_cpu_loss_safety = True 
            torch.cuda.synchronize() # Final sync before CPU transfer
            
            # Explicitly move to CPU and ensure contiguity to prevent segfaults
            cpu_logits = logits.float().cpu().contiguous()
            cpu_targets = targets.to(torch.int32).cpu().contiguous()
            cpu_logit_lengths = input_lengths.cpu().contiguous()
            cpu_target_lengths = target_lengths.cpu().contiguous()

            loss = rnnt_loss(
                logits=cpu_logits,
                targets=cpu_targets,
                logit_lengths=cpu_logit_lengths,
                target_lengths=cpu_target_lengths,
                blank=BLANK_IDX, reduction='mean'
            )
            if use_cpu_loss_safety: loss = loss.to(DEVICE)
            
            accum_steps = config['train']['accum_steps']
            scaler.scale(loss / accum_steps).backward()
            
            grad_norm = torch.tensor(0.0)
            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['grad_clipping'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            total_steps += 1

            # --- PERIODIC SAVE ---
            if total_steps % 1000 == 0:
                ckpt_path = os.path.join(config['paths']['checkpoint_dir'], "latest.pth")
                torch.save({
                    'epoch': epoch, 
                    'step': total_steps,
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'scaler_state_dict': scaler.state_dict()
                }, ckpt_path)
                print(f"Periodic checkpoint saved at step {total_steps}", flush=True)

            # --- REPAIRED METRICS BLOCK ---
            train_wer_sig, train_cer_sig = 0.0, 0.0
            spi, rtfx = 0.0, 0.0
            
            if batch_idx % config['monitoring']['log_interval'] == 0:
                spi = time.time() - step_start
                total_audio_sec = (audio_lengths.sum().item()) / 16000
                rtfx = total_audio_sec / spi
                
                model.eval()
                peek_num = min(2, waveforms.size(0))
                with torch.no_grad():
                    preds = model.greedy_decode(mel[:peek_num])
                    for i in range(peek_num):
                        ref = tokenizer.decode_ids(targets[i, :target_lengths[i]].tolist())
                        hyp = tokenizer.decode_ids(preds[i])
                        train_wer_sig += calculate_wer(ref, hyp)
                        train_cer_sig += calculate_wer(" ".join(list(ref)), " ".join(list(hyp)))
                    train_wer_sig /= peek_num
                    train_cer_sig /= peek_num
                model.train()
                
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f} | WER: {train_wer_sig:.1%} | SPI: {spi:.2f}s | RTFx: {rtfx:.1f}", flush=True)

                if wandb.run:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/grad_norm": grad_norm.item(),
                        "train/wer": train_wer_sig,
                        "train/rtfx": rtfx,
                        "train/spi": spi,
                        "epoch": epoch
                    })
                log_writer.writerow([epoch, batch_idx, f"{loss.item():.6f}", f"{grad_norm.item():.6f}", 
                                   f"{train_wer_sig:.4f}", f"{train_cer_sig:.4f}", "0.0", "0.0", f"{spi:.4f}", f"{rtfx:.2f}"])
                log_file.flush()

            if total_steps % config['monitoring']['validation_hook_interval'] == 0:
                print("\n --- Linguistic Peek ---")
                model.eval()
                with torch.no_grad():
                    for idx in random.sample(range(len(val_set)), 2):
                        v_wav, _, v_text, *_ = val_set[idx]
                        v_mel = mel_transform(v_wav.to(DEVICE)).squeeze(1)
                        v_mel = spec_norm(v_mel)
                        v_pred = tokenizer.decode_ids(model.greedy_decode(v_mel.unsqueeze(0))[0])
                        print(f"GT: {v_text.lower()}\nPR: {v_pred}\n" + "-"*30)
                model.train()
        
        # Validation Pass
        model.eval()
        val_loss, val_wer, val_cnt = 0, 0, 0
        with torch.no_grad():
            for v_wavs, v_tgts, v_alen, v_tlen in val_loader:
                v_wavs, v_tgts = v_wavs.to(DEVICE), v_tgts.to(DEVICE)
                v_mel = spec_norm(mel_transform(v_wavs).squeeze(1))
                v_logits = model(v_mel, v_tgts)
                
                v_l1 = torch.floor(((v_alen.float() / 160.0) - 3) / 2) + 1
                v_ilen = (torch.floor((v_l1 - 3) / 2) + 1).to(torch.int32)
                v_ilen = torch.clamp(v_ilen - 1, min=1)
                
                v_loss = rnnt_loss(v_logits.float(), v_tgts.to(torch.int32), v_ilen, v_tlen, blank=BLANK_IDX, reduction='mean')
                val_loss += v_loss.item()
                
                v_preds = model.greedy_decode(v_mel)
                for i, p in enumerate(v_preds):
                    val_wer += calculate_wer(tokenizer.decode_ids(v_tgts[i, :v_tlen[i]].tolist()), tokenizer.decode_ids(p))
                    val_cnt += 1
        
        print(f"--- Epoch {epoch} | Val Loss: {val_loss/len(val_loader):.4f} | WER: {val_wer/val_cnt:.2%} ---")
        scheduler.step()
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict()}, f"{config['paths']['checkpoint_dir']}/latest.pth")
    
    log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    train(limit_override=args.limit)
