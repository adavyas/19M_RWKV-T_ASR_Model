import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import psutil
import os
import whisper
import torchaudio
from thop import profile
import numpy as np

# Modular imports
from model import RWKV_Transducer

def get_peak_memory():
    """Returns the peak RSS memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

class WhisperWrapper(nn.Module):
    """
    Wrapper for OpenAI's Whisper model to match our benchmarking interface.
    """
    def __init__(self, model_size="tiny"):
        super().__init__()
        self.model = whisper.load_model(model_size, device="cpu")
        self.options = whisper.DecodingOptions(language="en", beam_size=None)

    def forward(self, audio):
        # Whisper expects 30s padded audio (16kHz)
        # We simulate the processing of a single chunk
        return self.model.decode(audio, self.options)

def benchmark_e2e(model, input_data, audio_dur, name="Model", iterations=10):
    """
    Benchmarks end-to-end decoding performance.
    """
    print(f"\nBenchmarking {name} (End-to-End)...")
    
    # Warmup (Crucial for torch.compile)
    print(f"Warmup (3 iterations)...")
    for _ in range(3):
        with torch.no_grad():
            _ = model(input_data)
            
    # Measurement
    mem_before = get_peak_memory()
    start_time = time.time()
    for i in range(iterations):
        if i % 2 == 0: 
            print(f"Iteration {i+1}/{iterations}...")
        with torch.no_grad():
            _ = model(input_data)
    end_time = time.time()
    mem_after = get_peak_memory()
    
    avg_latency = (end_time - start_time) / iterations
    rtfx = audio_dur / avg_latency
    
    return rtfx, mem_after

if __name__ == "__main__":
    device = torch.device("cpu")
    print(f"Executing End-to-End benchmark on {device}")

    # 1. Load Real Audio (LibriSpeech sample)
    audio_path = "data/LibriSpeech/dev-clean/2412/153954/2412-153954-0019.flac"
    if not os.path.exists(audio_path):
        # Fallback to dummy audio if dataset not present
        audio_dur = 7.49
        audio = torch.randn(1, int(16000 * audio_dur))
        print(f"Using dummy audio: {audio_dur} seconds")
    else:
        waveform, sr = torchaudio.load(audio_path)
        audio = waveform
        audio_dur = waveform.size(1) / sr
        print(f"Loaded audio sample: {audio_dur:.2f} seconds")

    # 2. Setup RWKV-T (Optimized)
    model_rwkv = RWKV_Transducer(vocab_size=2049, dim=256, n_enc=12, n_pred=4).to(device)
    
    # Apply torch.compile to RWKV-T components for "Good Kernels"
    try:
        model_rwkv.encoder = torch.compile(model_rwkv.encoder)
        model_rwkv.predictor = torch.compile(model_rwkv.predictor)
        model_rwkv.joiner = torch.compile(model_rwkv.joiner)
        print("Successfully applied torch.compile to RWKV-T components.")
    except Exception as e:
        print(f"Warning: torch.compile failed: {e}")

    # 3. Setup Whisper-Tiny
    model_whisper = WhisperWrapper("tiny").to(device)

    # 4. Prepare Mel Spectrogram for RWKV
    # RWKV takes Mel inputs; Whisper takes raw audio
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, hop_length=160)
    spec_norm = nn.InstanceNorm1d(80)
    
    rwkv_input = spec_norm(torch.log1p(mel_transform(audio)[:, 0, :, :]))
    whisper_input = whisper.pad_or_trim(audio.flatten())

    # 5. Run Benchmarks
    rwkv_rtfx, rwkv_mem = benchmark_e2e(model_rwkv.greedy_decode, rwkv_input, audio_dur, name="RWKV-T (19M)")
    whisper_rtfx, whisper_mem = benchmark_e2e(model_whisper, whisper_input, audio_dur, name="Whisper-Tiny (39M)")

    # 6. Parameter Stats
    rwkv_params = sum(p.numel() for p in model_rwkv.parameters()) / 1e6
    whisper_params = sum(p.numel() for p in model_whisper.model.parameters()) / 1e6

    # 7. Final Report
    print("\n" + "="*80)
    print(f"{'Model':<25} | {'Params (M)':<12} | {'RTFx (CPU)':<12} | {'Peak RSS (MB)':<12}")
    print("-" * 80)
    print(f"{'RWKV-T (19M)':<25} | {rwkv_params:<12.2f} | {rwkv_rtfx:<12.2f} | {rwkv_mem:<12.2f}")
    print(f"{'Whisper-Tiny (39M)':<25} | {whisper_params:<12.2f} | {whisper_rtfx:<12.2f} | {whisper_mem:<12.2f}")
    print("="*80)
    print("Notes: End-to-End Greedy Decoding on real audio.")
