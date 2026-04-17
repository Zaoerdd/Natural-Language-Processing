import argparse
import json
import zipfile
from pathlib import Path


def iter_zip_members(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in sorted(zf.infolist(), key=lambda x: x.filename):
            if member.is_dir():
                continue
            if member.filename.startswith("__MACOSX/"):
                continue
            yield member


def extract_member_to_text(zip_path, member, output_root):
    output_path = output_root / Path(member.filename)
    output_path = output_path.with_suffix(".txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    article_count = 0
    char_count = 0
    with zipfile.ZipFile(zip_path, "r") as zf, zf.open(member, "r") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for raw_line in src:
            raw_line = raw_line.decode("utf-8").strip()
            if not raw_line:
                continue
            record = json.loads(raw_line)
            title = str(record.get("title", "")).strip()
            text = str(record.get("text", "")).strip()
            pieces = [piece for piece in (title, text) if piece]
            if not pieces:
                continue
            article = "\n\n".join(pieces)
            dst.write(article)
            dst.write("\n<|endoftext|>\n")
            article_count += 1
            char_count += len(article)

    return output_path, article_count, char_count


parser = argparse.ArgumentParser(
    description="Extract and preprocess Chinese Wikipedia JSONL shards into plain-text files."
)
parser.add_argument(
    "--input_zip",
    type=str,
    required=True,
    help="Path to wiki_zh_2019.zip",
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory to store extracted .txt files.",
)
parser.add_argument(
    "--limit_files",
    type=int,
    default=None,
    help="Optional limit for debugging.",
)
parser.add_argument(
    "--summary_json",
    type=str,
    default=None,
    help="Optional path to save extraction statistics as JSON.",
)
args = parser.parse_args()

zip_path = Path(args.input_zip)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

summary = {
    "input_zip": str(zip_path.resolve()),
    "output_dir": str(output_dir.resolve()),
    "file_count": 0,
    "article_count": 0,
    "character_count": 0,
    "files": [],
}

for idx, member in enumerate(iter_zip_members(zip_path), start=1):
    if args.limit_files is not None and idx > args.limit_files:
        break
    output_path, article_count, char_count = extract_member_to_text(zip_path, member, output_dir)
    rel_path = output_path.relative_to(output_dir).as_posix()
    summary["file_count"] += 1
    summary["article_count"] += article_count
    summary["character_count"] += char_count
    summary["files"].append(
        {
            "member": member.filename,
            "output_path": rel_path,
            "article_count": article_count,
            "character_count": char_count,
        }
    )
    print(
        f"[{summary['file_count']:04d}] {member.filename} -> {rel_path} "
        f"articles={article_count:,} chars={char_count:,}"
    )

print(
    f"\nFinished extraction: files={summary['file_count']:,}, "
    f"articles={summary['article_count']:,}, chars={summary['character_count']:,}"
)

if args.summary_json:
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved extraction summary to: {summary_path}")
