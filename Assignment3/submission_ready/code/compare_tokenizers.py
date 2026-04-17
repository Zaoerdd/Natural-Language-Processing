import argparse
import json
import sys

import tiktoken
from tokenizers import Tokenizer


DEFAULT_SAMPLE_TEXT = "\u592a\u9633\u7167\u5e38\u5347\u8d77\u3002"


def safe_print(*parts):
    text = " ".join(str(part) for part in parts)
    sys.stdout.buffer.write(text.encode("utf-8", errors="backslashreplace") + b"\n")


parser = argparse.ArgumentParser(
    description="Compare a trained Chinese tokenizer with the original GPT-2 tokenizer."
)
parser.add_argument(
    "--tokenizer",
    type=str,
    default="wikizh_tokenizer_from_scratch.json",
    help="Path to the trained tokenizer JSON file.",
)
parser.add_argument(
    "--text",
    type=str,
    default=DEFAULT_SAMPLE_TEXT,
    help="Sample text to tokenize.",
)
parser.add_argument(
    "--output_json",
    type=str,
    default=None,
    help="Optional path to save the comparison result as JSON.",
)
args = parser.parse_args()

# Load the tokenizer trained from scratch
tokenizer_from_scratch = Tokenizer.from_file(args.tokenizer)

# Original GPT-2 tokenizer via tiktoken
tokenizer_gpt2_original = tiktoken.get_encoding("gpt2")

trained_encoding = tokenizer_from_scratch.encode(args.text)
gpt2_ids = tokenizer_gpt2_original.encode(
    args.text, allowed_special={"<|endoftext|>"}
)

result = {
    "text": args.text,
    "trained_tokenizer_ids": trained_encoding.ids,
    "trained_tokenizer_pieces": trained_encoding.tokens,
    "trained_tokenizer_token_count": len(trained_encoding.ids),
    "gpt2_tokenizer_ids": gpt2_ids,
    "gpt2_tokenizer_pieces_repr": [
        repr(tokenizer_gpt2_original.decode_single_token_bytes(tid)) for tid in gpt2_ids
    ],
    "gpt2_tokenizer_token_count": len(gpt2_ids),
}

safe_print("Sample text:", args.text)
safe_print("Trained tokenizer token count:", result["trained_tokenizer_token_count"])
safe_print("Trained tokenizer IDs        :", result["trained_tokenizer_ids"])
safe_print("Trained tokenizer pieces     :", result["trained_tokenizer_pieces"])
safe_print("GPT-2 tokenizer token count  :", result["gpt2_tokenizer_token_count"])
safe_print("GPT-2 tokenizer IDs          :", result["gpt2_tokenizer_ids"])
safe_print("GPT-2 token byte pieces      :", result["gpt2_tokenizer_pieces_repr"])

if args.output_json:
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    safe_print(f"Saved comparison JSON to: {args.output_json}")
