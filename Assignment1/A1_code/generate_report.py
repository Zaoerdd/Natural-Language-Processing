from pathlib import Path

from a1_nn_utils import (
    ExperimentConfig,
    compare_vocab_sizes,
    dataset_summary,
    load_jsonl,
    make_train_val_split,
    run_all_experiments,
)


def to_table_text(df) -> str:
    return df.to_string(index=False)


def build_report_text() -> str:
    config = ExperimentConfig(
        batch_size=64,
        embed_dim=128,
        hidden_dims=(128, 64),
        dropout=0.2,
        learning_rate=2e-3,
        epochs=10,
        min_freq=2,
        val_size=0.1,
        seed=42,
        device="cpu",
    )

    full_train_examples = load_jsonl(config.train_path)
    train_examples, _ = make_train_val_split(
        full_train_examples,
        val_size=config.val_size,
        seed=config.seed,
    )
    summary_df = dataset_summary(config.train_path, config.test_path)
    vocab_df = compare_vocab_sizes(train_examples, min_freq=config.min_freq)
    _, results_df = run_all_experiments(config=config, verbose=False)

    best_row = results_df.iloc[0]

    report = f"""# Assignment 1 Report: Neural Network-based Text Classification

## 1. Task Overview

This assignment trains a neural network-based text classifier on the Chinese humor detection dataset.
I implemented two tokenizers:

- `basic_char`: keeps only Chinese characters and treats each character as one token.
- `advanced_jieba`: segments Chinese spans into words with `jieba`, while preserving English words, numbers, and punctuations.

The classifier is based on `nn.EmbeddingBag` and a fully-connected network with two hidden layers (`128 -> 64`), which satisfies the assignment requirement.

## 2. Dataset Summary

```
{to_table_text(summary_df)}
```

The training data is clearly imbalanced, with fewer positive humor examples than negative examples. This makes precision/recall/F1 more informative than accuracy alone.

## 3. Vocabulary Comparison

```
{to_table_text(vocab_df)}
```

The advanced tokenizer produces a much larger vocabulary because it keeps richer token patterns such as Chinese multi-character words, English fragments, numbers, and punctuations.

## 4. Test Results

```
{to_table_text(results_df.round(4))}
```

Best final model on the test set in this run: **{best_row['tokenizer']}**

- Accuracy: {best_row['accuracy']:.4f}
- Precision: {best_row['precision']:.4f}
- Recall: {best_row['recall']:.4f}
- F1: {best_row['f1']:.4f}

## 5. Discussion

In this experiment, the advanced tokenizer achieved slightly better overall test performance, while also substantially increasing vocabulary size.
This suggests that richer tokenization can help the classifier capture additional useful patterns, but it comes with a larger vocabulary and therefore higher model complexity.
However, the advanced tokenizer remains useful because it captures more linguistic structure and special patterns than pure character splitting.

## 6. Reproducibility

- Random seed: 42
- Optimizer: Adam
- Learning rate: 0.002
- Batch size: 64
- Epochs: 10
- Validation split: 10% of the provided training set
- Device: CPU
"""
    return report


def save_pdf(report_text: str, out_path: Path) -> None:
    import textwrap

    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    page_width, page_height = A4
    pdf = canvas.Canvas(str(out_path), pagesize=A4)
    pdf.setFont("Courier", 9)

    x = 40
    y = page_height - 40
    line_height = 12

    for raw_line in report_text.splitlines():
        wrapped_lines = textwrap.wrap(raw_line, width=110) or [" "]
        for line in wrapped_lines:
            if y < 40:
                pdf.showPage()
                pdf.setFont("Courier", 9)
                y = page_height - 40
            pdf.drawString(x, y, line)
            y -= line_height

    pdf.save()


def main() -> None:
    report_text = build_report_text()
    md_path = Path("A1_report.md")
    md_path.write_text(report_text, encoding="utf-8")
    print(f"Wrote {md_path.resolve()}")

    try:
        pdf_path = Path("A1_report.pdf")
        save_pdf(report_text, pdf_path)
        print(f"Wrote {pdf_path.resolve()}")
    except Exception as exc:
        print(f"Skipping PDF export: {exc}")


if __name__ == "__main__":
    main()
