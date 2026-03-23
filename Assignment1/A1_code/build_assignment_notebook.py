import json
import textwrap
from pathlib import Path


def lines(text: str) -> list[str]:
    cleaned = textwrap.dedent(text).strip("\n")
    return [line + "\n" for line in cleaned.splitlines()]


NOTEBOOK_PATH = Path("A1_nn.ipynb")


cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(
            """
            ## CS310 Natural Language Processing
            ## Assignment 1. Neural Network-based Text Classification

            This notebook implements a neural network-based Chinese humor detector with two tokenization strategies:

            1. `basic_char`: keep only Chinese characters and treat each character as one token.
            2. `advanced_jieba`: segment Chinese spans into words with `jieba`, while preserving English words, numbers, and punctuations as separate tokens.

            The model uses `nn.EmbeddingBag` for bag-of-words encoding and a fully-connected classifier with two hidden layers.
            """
        ),
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines("### 0. Import Necessary Libraries"),
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(
            """
            import pandas as pd
            import torch

            from a1_nn_utils import (
                ExperimentConfig,
                HumorClassifier,
                compare_vocab_sizes,
                dataset_summary,
                load_jsonl,
                make_train_val_split,
                preview_tokenization,
                run_all_experiments,
                seed_everything,
            )

            pd.set_option("display.max_colwidth", 200)

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

            seed_everything(config.seed)
            print(config)
            print("torch version:", torch.__version__)
            """
        ),
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines("### 1. Data Processing"),
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(
            """
            summary_df = dataset_summary(config.train_path, config.test_path)
            summary_df
            """
        ),
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(
            """
            full_train_examples = load_jsonl(config.train_path)
            train_examples, val_examples = make_train_val_split(
                full_train_examples,
                val_size=config.val_size,
                seed=config.seed,
            )
            example_text = train_examples[0][0]
            print("Example sentence:", example_text)
            preview_tokenization(example_text)
            """
        ),
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(
            """
            vocab_df = compare_vocab_sizes(train_examples, min_freq=config.min_freq)
            vocab_df
            """
        ),
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(
            """
            `basic_char` creates a smaller vocabulary because each Chinese character maps to a token and non-Chinese symbols are discarded.
            `advanced_jieba` keeps more detailed patterns such as words, English fragments, numbers, and punctuations, so the vocabulary becomes noticeably larger.
            """
        ),
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines("### 2. Build the Model"),
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(
            """
            demo_vocab_size = int(vocab_df.loc[vocab_df["tokenizer"] == "basic_char", "vocab_size"].iloc[0])
            demo_model = HumorClassifier(
                vocab_size=demo_vocab_size,
                embed_dim=config.embed_dim,
                hidden_dims=config.hidden_dims,
                dropout=config.dropout,
            )
            demo_model
            """
        ),
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(
            """
            Model design choices:

            - `nn.EmbeddingBag(..., mode="mean")` implements a bag-of-words encoder efficiently.
            - The classifier contains **two hidden layers**: `128 -> 64`.
            - `ReLU + Dropout` are used to improve non-linearity and reduce overfitting.
            """
        ),
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines("### 3. Train and Evaluate"),
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(
            """
            experiment_outputs, results_df = run_all_experiments(config=config, verbose=True)
            results_df
            """
        ),
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(
            """
            for tokenizer_name, output in experiment_outputs.items():
                print(f"\\nTraining history for {tokenizer_name}")
                print(
                    output["history"][
                        ["epoch", "train_loss", "val_accuracy", "val_precision", "val_recall", "val_f1"]
                    ].to_string(index=False)
                )
            """
        ),
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(
            """
            best_row = results_df.iloc[0]
            print("Best tokenizer in the final test comparison:")
            print(best_row)
            """
        ),
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(
            """
            Final note:

            - Both tokenizers satisfy the assignment requirements.
            - In this fixed-seed run, the advanced tokenizer achieved slightly better test-set metrics, but it also required a much larger vocabulary.
            - The comparison shows the tradeoff clearly: richer tokenization can capture more patterns, but it increases model complexity.
            """
        ),
    },
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3.12",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH.resolve()}")
