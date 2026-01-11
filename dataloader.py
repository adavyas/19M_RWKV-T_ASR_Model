import os
import torch
import torchaudio
from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import DataLoader, Subset, IterableDataset
import random
from datasets import load_dataset
import platform

class LibriCollate:
    """
    Collate function for LibriSpeech audio data.
    Pads waveforms and transcripts for batch processing.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        waveforms, transcripts = [], []
        audio_lengths, label_lengths = [], []

        for wav, _, transcript, *_ in batch:
            # waveforms are expected to be 16kHz
            waveforms.append(wav.squeeze(0))
            ids = self.tokenizer.EncodeAsIds(transcript.lower())
            targets = torch.tensor(ids, dtype=torch.long)
            transcripts.append(targets)
            audio_lengths.append(wav.size(-1))
            label_lengths.append(len(ids))

        waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True).unsqueeze(1)
        transcripts = torch.nn.utils.rnn.pad_sequence(transcripts, batch_first=True, padding_value=0)
        
        langs = ["en"] * len(batch)
        
        return waveforms, transcripts, torch.tensor(audio_lengths, dtype=torch.int32), torch.tensor(label_lengths, dtype=torch.int32), langs

class HFStreamingTextDataset(IterableDataset):
    """
    Streams text data from Hugging Face datasets for Predictor (LM) pre-training.
    Supports multi-process loading and dataset mixing.
    """
    def __init__(self, tokenizer, ds_configs):
        self.tokenizer = tokenizer
        self.datasets = []
        for path, name, split, text_key in ds_configs:
            try:
                ds = load_dataset(path, name, split=split, streaming=True)
                if "audio" in ds.features:
                    ds = ds.remove_columns(["audio"])
                self.datasets.append((ds, text_key))
            except Exception as e:
                print(f"Warning: Could not stream {path} ({name}): {e}")

    def __iter__(self):
        if not self.datasets:
            return

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            my_datasets = self.datasets
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            my_datasets = [d for i, d in enumerate(self.datasets) if i % num_workers == worker_id]
            if not my_datasets:
                my_datasets = [self.datasets[worker_id % len(self.datasets)]]

        while True:
            random.shuffle(my_datasets)
            for ds, text_key in my_datasets:
                for ex in ds:
                    text = ex[text_key]
                    if not text or len(text) < 5: 
                        continue
                    if isinstance(text, (list, tuple)): 
                        text = text[0]
                    yield text.lower(), "en"

class LibriTextCollate:
    """
    Collate function for text-only data (Predictor pre-training).
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        transcripts, label_lengths = [], []
        langs = []
        for text, lang in batch:
            ids = self.tokenizer.EncodeAsIds(text)
            targets = torch.tensor(ids, dtype=torch.long)
            transcripts.append(targets)
            label_lengths.append(len(ids))
            langs.append(lang)
        
        transcripts = torch.nn.utils.rnn.pad_sequence(transcripts, batch_first=True, padding_value=0)
        # Return dummy audio tensors
        return torch.zeros((len(batch), 1, 1, 1)), transcripts, torch.zeros(len(batch), dtype=torch.long), torch.tensor(label_lengths, dtype=torch.long), langs

def get_dataloaders(mode, tokenizer, batch_size, num_workers, limit_samples=None, max_audio_duration=15.0):
    """
    Returns appropriate dataloaders for the specified training mode.
    """
    if mode == "predictor":
        # Predictor mode uses streaming text datasets
        configs = [
            ("sentence-transformers/parallel-sentences-opensubtitles", "en-de", "train", "english"),
            ("librispeech_asr", "clean", "train.100", "text"),
            ("wikitext", "wikitext-103-v1", "train", "text")
        ]
        
        train_set = HFStreamingTextDataset(tokenizer, configs)
        
        # Darwin (Mac) stability fix for streaming
        actual_workers = num_workers if platform.system() != "Darwin" else 0
            
        train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=LibriTextCollate(tokenizer), num_workers=actual_workers)
        
        val_set = HFStreamingTextDataset(tokenizer, [configs[1]])
        val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=LibriTextCollate(tokenizer), num_workers=0)
        
        return train_loader, val_loader, train_set, val_set

    # Encoder or Joint mode uses LibriSpeech
    try:
        train_set = LIBRISPEECH("./data", url="train-clean-100", download=True)
    except Exception as e:
        print(f"Error loading LibriSpeech: {e}")
        return None, None, None, None

    if limit_samples:
        train_set = Subset(train_set, range(min(limit_samples, len(train_set))))
    
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=LibriCollate(tokenizer),
        num_workers=num_workers,
        pin_memory=(torch.cuda.is_available())
    )

    val_set = LIBRISPEECH("./data", url="dev-clean", download=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=LibriCollate(tokenizer), num_workers=num_workers)

    return train_loader, val_loader, train_set, val_set
