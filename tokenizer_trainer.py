import os
import torchaudio
from torchaudio.datasets import LIBRISPEECH
import sentencepiece as spm

def train_tokenizer():
    print("Preparing LibriSpeech-100 transcripts for tokenizer training...")
    
    # We only need the transcripts, so we use 'train-clean-100'
    # download=True will download if not present
    dataset = LIBRISPEECH("./data", url="train-clean-100", download=True)
    
    transcripts_path = "libri_transcripts.txt"
    
    # Extract all transcripts to a temporary file
    if not os.path.exists(transcripts_path):
        print("Extracting transcripts...")
        with open(transcripts_path, "w", encoding="utf-8") as f:
            for i in range(len(dataset)):
                # dataset[i] returns (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)
                _, _, transcript, _, _, _ = dataset[i]
                f.write(transcript.lower() + "\n")
    
    print("Training BPE tokenizer (Vocab size: 1024)...")
    # Training parameters
    # --input: the file with transcripts
    # --model_prefix: name of the output model file
    # --vocab_size: 1024
    # --model_type: bpe
    # --character_coverage: 1.0 (Latin alphabet only basically)
    # --pad_id: 0 (Will be used as <blank> in RNN-T)
    # --unk_id: 1
    # --bos_id: 2
    # --eos_id: 3
    # --user_defined_symbols: <blank> (ensure it's at index 0)
    
    # Note: SentencePiece default:
    # <unk> = 0
    # We let the model add <blank> at index 2048 (vocab_size).
    
    spm.SentencePieceTrainer.train(
        input=transcripts_path,
        model_prefix='bpe',
        vocab_size=2048,
        model_type='bpe',
        character_coverage=1.0,
        unk_id=0,
        bos_id=-1, # Disable default bos/eos to match RNN-T needs
        eos_id=-1,
        pad_id=-1,
    )
    
    print("BPE tokenizer training complete. Files saved: bpe.model, bpe.vocab")
    
    # Cleanup
    if os.path.exists(transcripts_path):
        os.remove(transcripts_path)

if __name__ == "__main__":
    if not os.path.exists("./data"):
        os.makedirs("./data")
    train_tokenizer()
