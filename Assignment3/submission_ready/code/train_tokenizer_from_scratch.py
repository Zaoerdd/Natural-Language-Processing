import argparse
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

parser = argparse.ArgumentParser(
    description="Train a BPE tokenizer on Chinese text data from scratch."
)
parser.add_argument(
    "--input", type=str, required=True, help="Path to the input Chinese text file."
)
parser.add_argument(
    "--vocab_size", type=int, default=52000, help="Vocabulary size."
)
parser.add_argument(
    "--pre_tokenizer", type=str, choices=["Whitespace", "ByteLevel"], default="Whitespace", help="Pre-tokenizer to use."
)
parser.add_argument(
    "--min_freq", type=int, default=2, help="Minimum frequency for a word to be included in the vocabulary."
)
parser.add_argument(
    "--output", type=str, default="wikizh_tokenizer_from_scratch.json", help="Path to save the trained tokenizer."
)
args = parser.parse_args()


def collect_training_files(input_path):
    path = Path(input_path)
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        files = sorted(str(p) for p in path.rglob("*.txt"))
        if not files:
            raise ValueError(f"No .txt files found under {input_path}")
        return files
    raise ValueError(f"Input path does not exist: {input_path}")

# Initialize a BPE tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="<|endoftext|>"))

# Pre-tokenize 
if args.pre_tokenizer == "Whitespace":
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
elif args.pre_tokenizer == "ByteLevel":
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

# Train on your chinese text file
trainer = trainers.BpeTrainer(
    vocab_size=args.vocab_size, 
    special_tokens=["<|endoftext|>"],
    min_frequency=args.min_freq,
)
training_files = collect_training_files(args.input)
print(f"Training tokenizer on {len(training_files)} text file(s)")
tokenizer.train(training_files, trainer)

# Save the tokenizer
tokenizer.save(args.output)
print(f"Saved tokenizer to: {args.output}")
