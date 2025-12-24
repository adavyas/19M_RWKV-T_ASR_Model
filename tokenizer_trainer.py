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
    
    # Check if transcripts file exists and is not empty
    if os.path.exists(transcripts_path) and os.path.getsize(transcripts_path) == 0:
        print(f"Detected empty {transcripts_path}. Deleting and re-extracting...")
        os.remove(transcripts_path)

    # Extract all transcripts to a temporary file
    if not os.path.exists(transcripts_path):
        print("Extracting transcripts (Fast Path - searching for .trans.txt files)...")
        # Find all .trans.txt files in the data directory
        # Torchaudio downloads to {path}/LibriSpeech/{url}
        search_path = os.path.join(dataset._path, "LibriSpeech", dataset._url)
        print(f"Searching for transcripts in: {search_path}")
        
        trans_files = []
        if os.path.exists(search_path):
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith(".trans.txt"):
                        trans_files.append(os.path.join(root, file))
        
        if not trans_files:
            print(f"No .trans.txt files found in {search_path}. Checking {dataset._path}...")
            # Fallback search in entire data root
            for root, dirs, files in os.walk(dataset._path):
                for file in files:
                    if file.endswith(".trans.txt") and dataset._url in root:
                        trans_files.append(os.path.join(root, file))
        
        if not trans_files:
            print("No .trans.txt files found. Falling back to slow extraction...")
            # Fallback to slow method if files aren't found for some reason
            with open(transcripts_path, "w", encoding="utf-8") as f:
                for i in range(len(dataset)):
                    if i % 1000 == 0: print(f"Processing sample {i}/{len(dataset)}...")
                    _, _, transcript, _, _, _ = dataset[i]
                    f.write(transcript.lower() + "\n")
        else:
            print(f"Found {len(trans_files)} transcription files. Merging...")
            with open(transcripts_path, "w", encoding="utf-8") as out_f:
                for tf in trans_files:
                    with open(tf, "r") as in_f:
                        for line in in_f:
                            # LibriSpeech trans files are "ID TEXT..."
                            content = " ".join(line.strip().split()[1:])
                            out_f.write(content.lower() + "\n")
        print("Transcript extraction complete.")
    else:
        print(f"Found existing {transcripts_path}, skipping extraction.")
    
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
