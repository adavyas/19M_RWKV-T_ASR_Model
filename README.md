# RWKV-T: Efficient Recurrent ASR (19M Parameters)

This repository contains a Proof-of-Concept (PoC) implementation of **RWKV-T**, an Automatic Speech Recognition (ASR) model based on the RWKV-7 architecture. RWKV-T combines the efficiency of recurrent neural networks with the performance of state-of-the-art Transducers.

## Key Features

- **RWKV-7 Architecture**: Leverages the latest "Goose" recurrence (Generalized Delta Rule) for efficient sequence modeling.
- **19M Parameter Scale**: Optimized for edge deployment and fast CPU inference.
- **O(1) Inference Memory**: Unlike Transformers, RWKV-T maintains a fixed-size hidden state (approx. 200KB), making it ideal for streaming and memory-constrained devices.
- **Modular Design**: Cleanly separated components for the Encoder, Predictor, and Joiner.
- **Multi-Stage Training**: Supports Encoder pre-training (CTC), Predictor pre-training (Language Modeling), and Joint RNN-T training.

## Architecture Highlights

RWKV-T utilizes a **RNN-Transducer (RNN-T)** framework:
- **Encoder**: 12 RWKV-7 blocks with a 4x temporal subsampling frontend.
- **Predictor**: 4 RWKV-7 blocks acting as a stateful language model.
- **Joiner**: A non-linear feedforward network that fuses acoustic and linguistic representations.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training
Configure your hyperparameters in `config.yaml` and run:
```bash
# Joint RNN-T training
python train.py --mode joint

# Predictor-only pre-training (Language Modeling)
python train.py --mode predictor
```

### Benchmarking
Compare RWKV-T against Whisper-Tiny on your local CPU:
```bash
python benchmark_models.py
```

## Repository Structure

- `model.py`: Core RWKV-7 Transducer definitions.
- `train.py`: Main training loop with multi-mode support and W&B integration.
- `dataloader.py`: Robust data pipelines for LibriSpeech and streaming HF datasets.
- `utils.py`: ASR metrics (WER/CER) and helper utilities.
- `config.yaml`: Central configuration for model and training.
- `benchmark_models.py`: Qualitative and quantitative comparison script.

## Performance Benchmarks (CPU)

On a standard Mac CPU, the 19M RWKV-T achieves highly competitive results after kernel optimization via `torch.compile`:

| Model | Params (M) | RTFx (CPU) | Peak RSS (MB) |
| :--- | :--- | :--- | :--- |
| **RWKV-T (19M)** | 19.3 | **~35x** | ~1100 |
| **Whisper-Tiny** | 37.2 | ~48x | ~1200 |

*Note: RTFx (Real-Time Factor) indicates how many seconds of audio are processed per second of compute.*

## Contributing
This is an active research project. Feedback and contributions to optimized kernels (Triton/C++) are highly encouraged!
