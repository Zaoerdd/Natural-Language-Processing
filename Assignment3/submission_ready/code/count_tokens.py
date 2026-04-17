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


def collect_token_files(input_path):
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("*.npy"))
        if not files:
            raise ValueError(f"No .npy files found under {input_path}")
        return files
    raise ValueError(f"Input path does not exist: {input_path}")


parser = argparse.ArgumentParser(
    description="Count tokens in a text corpus or token shard directory."
)
parser.add_argument("--input", type=str, required=True, help="Path to .txt/.npy file or directory.")
parser.add_argument(
    "--tokenizer",
    type=str,
    default=None,
    help="Tokenizer JSON path. Required when counting raw text files.",
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["auto", "text", "tokens"],
    default="auto",
    help="Whether the input contains raw text or tokenized .npy shards.",
)
parser.add_argument(
    "--output_json",
    type=str,
    default=None,
    help="Optional JSON path for saving per-file token counts.",
)
args = parser.parse_args()

input_path = Path(args.input)
mode = args.mode
if mode == "auto":
    if input_path.is_file() and input_path.suffix.lower() == ".npy":
        mode = "tokens"
    elif input_path.is_dir() and any(input_path.rglob("*.npy")):
        mode = "tokens"
    else:
        mode = "text"

summary = {
    "input": str(input_path.resolve()),
    "mode": mode,
    "total_tokens": 0,
    "file_count": 0,
    "files": [],
}

if mode == "text":
    if not args.tokenizer:
        raise ValueError("--tokenizer is required when counting raw text files")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    files = collect_text_files(input_path)
    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        token_count = len(tokenizer.encode(text).ids)
        summary["file_count"] += 1
        summary["total_tokens"] += token_count
        summary["files"].append(
            {"path": str(file_path), "token_count": token_count}
        )
        print(f"{file_path}: {token_count:,}")
else:
    files = collect_token_files(input_path)
    for file_path in files:
        token_count = int(np.load(file_path, mmap_mode="r").shape[0])
        summary["file_count"] += 1
        summary["total_tokens"] += token_count
        summary["files"].append(
            {"path": str(file_path), "token_count": token_count}
        )
        print(f"{file_path}: {token_count:,}")

print(
    f"\nCounted {summary['total_tokens']:,} tokens across "
    f"{summary['file_count']:,} file(s)"
)

if args.output_json:
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved token count summary to: {output_path}")
