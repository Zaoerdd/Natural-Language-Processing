import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def collect_text_files(input_path):
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("*.txt"))
        if not files:
            raise ValueError(f"No .txt files found under {input_path}")
        return files
    raise ValueError(f"Input path does not exist: {input_path}")


parser = argparse.ArgumentParser(
    description="Encode extracted text shards into token-id numpy arrays."
)
parser.add_argument("--input", type=str, required=True, help="Path to .txt file or directory.")
parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer JSON file.")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .npy shards.")
parser.add_argument(
    "--limit_files",
    type=int,
    default=None,
    help="Optional file limit for debugging.",
)
args = parser.parse_args()

input_path = Path(args.input)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

tokenizer = Tokenizer.from_file(args.tokenizer)
vocab_size = tokenizer.get_vocab_size()
dtype = np.uint16 if vocab_size < 65536 else np.uint32
files = collect_text_files(input_path)
base_dir = input_path if input_path.is_dir() else input_path.parent

manifest = {
    "input": str(input_path.resolve()),
    "tokenizer": str(Path(args.tokenizer).resolve()),
    "output_dir": str(output_dir.resolve()),
    "vocab_size": vocab_size,
    "dtype": str(np.dtype(dtype)),
    "file_count": 0,
    "total_tokens": 0,
    "files": [],
}

for idx, file_path in enumerate(files, start=1):
    if args.limit_files is not None and idx > args.limit_files:
        break

    text = file_path.read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text).ids
    array = np.asarray(token_ids, dtype=dtype)

    rel_path = file_path.relative_to(base_dir).with_suffix(".npy")
    output_path = output_dir / rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array, allow_pickle=False)

    token_count = int(array.shape[0])
    manifest["file_count"] += 1
    manifest["total_tokens"] += token_count
    manifest["files"].append(
        {
            "relative_path": rel_path.as_posix(),
            "source_text_path": str(file_path.resolve()),
            "token_count": token_count,
        }
    )
    print(
        f"[{manifest['file_count']:04d}] {file_path} -> {output_path} "
        f"tokens={token_count:,}"
    )

manifest_path = output_dir / "manifest.json"
with manifest_path.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print(
    f"\nFinished tokenization: files={manifest['file_count']:,}, "
    f"tokens={manifest['total_tokens']:,}"
)
print(f"Saved manifest to: {manifest_path}")
