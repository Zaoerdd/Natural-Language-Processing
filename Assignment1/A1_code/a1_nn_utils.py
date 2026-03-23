import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import jieba
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader


SEED = 42
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
ADV_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z]+|\d+|[^\w\s]")


@dataclass
class ExperimentConfig:
    train_path: str = "train.jsonl"
    test_path: str = "test.jsonl"
    batch_size: int = 64
    embed_dim: int = 128
    hidden_dims: tuple[int, int] = (128, 64)
    dropout: float = 0.2
    learning_rate: float = 2e-3
    epochs: int = 10
    min_freq: int = 2
    val_size: float = 0.1
    seed: int = SEED
    device: str = "cpu"


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_jsonl(path: str | Path) -> list[tuple[str, int]]:
    examples = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            label = obj["label"][0] if isinstance(obj["label"], list) else obj["label"]
            examples.append((obj["sentence"], int(label)))
    return examples


def basic_char_tokenizer(text: str) -> list[str]:
    return [char for char in text if CJK_RE.fullmatch(char)]


def advanced_tokenizer(text: str) -> list[str]:
    tokens = []
    for chunk in ADV_PATTERN.findall(text):
        if chunk and all(CJK_RE.fullmatch(char) for char in chunk):
            tokens.extend(token for token in jieba.lcut(chunk) if token.strip())
        else:
            tokens.append(chunk)
    return tokens


TOKENIZERS = {
    "basic_char": basic_char_tokenizer,
    "advanced_jieba": advanced_tokenizer,
}


def preview_tokenization(text: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tokenizer": list(TOKENIZERS.keys()),
            "tokens": [TOKENIZERS[name](text) for name in TOKENIZERS],
        }
    )


def dataset_summary(train_path: str | Path = "train.jsonl", test_path: str | Path = "test.jsonl") -> pd.DataFrame:
    rows = []
    for split_name, path in [("train", train_path), ("test", test_path)]:
        data = load_jsonl(path)
        labels = Counter(label for _, label in data)
        rows.append(
            {
                "split": split_name,
                "num_examples": len(data),
                "label_0": labels.get(0, 0),
                "label_1": labels.get(1, 0),
                "positive_ratio": round(labels.get(1, 0) / len(data), 4),
            }
        )
    return pd.DataFrame(rows)


def make_train_val_split(
    train_examples: list[tuple[str, int]], val_size: float = 0.1, seed: int = SEED
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    labels = [label for _, label in train_examples]
    return train_test_split(train_examples, test_size=val_size, stratify=labels, random_state=seed)


def build_vocab(
    texts: list[str], tokenizer, min_freq: int = 1
) -> tuple[dict[str, int], Counter]:
    counter = Counter()
    for text in texts:
        counter.update(tokenizer(text))
    vocab = {"<unk>": 0}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab, counter


def compare_vocab_sizes(
    train_examples: list[tuple[str, int]],
    min_freq: int = 1,
) -> pd.DataFrame:
    rows = []
    texts = [text for text, _ in train_examples]
    for name, tokenizer in TOKENIZERS.items():
        vocab, counter = build_vocab(texts, tokenizer, min_freq=min_freq)
        rows.append(
            {
                "tokenizer": name,
                "min_freq": min_freq,
                "vocab_size": len(vocab),
                "observed_tokens": sum(counter.values()),
                "avg_tokens_per_example": round(sum(counter.values()) / len(texts), 2),
            }
        )
    return pd.DataFrame(rows)


def encode_text(text: str, tokenizer, vocab: dict[str, int]) -> list[int]:
    token_ids = [vocab.get(token, 0) for token in tokenizer(text)]
    return token_ids or [0]


def make_collate_fn(tokenizer, vocab: dict[str, int]):
    def collate_fn(batch):
        labels = []
        flat_token_ids = []
        offsets = [0]
        total = 0
        for text, label in batch:
            token_ids = encode_text(text, tokenizer, vocab)
            flat_token_ids.extend(token_ids)
            labels.append(label)
            total += len(token_ids)
            offsets.append(total)

        return (
            torch.tensor(flat_token_ids, dtype=torch.long),
            torch.tensor(offsets[:-1], dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    return collate_fn


class HumorClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], num_classes),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.embedding.weight)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, text: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text, offsets)
        return self.classifier(embedded)


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def evaluate_model(
    model: HumorClassifier,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: str = "cpu",
) -> dict[str, float]:
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for text, offsets, labels in data_loader:
            text = text.to(device)
            offsets = offsets.to(device)
            labels = labels.to(device)
            logits = model(text, offsets)
            loss = loss_fn(logits, labels)
            preds = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / total_examples
    return metrics


def train_single_experiment(
    tokenizer_name: str,
    config: ExperimentConfig | None = None,
    verbose: bool = True,
) -> dict:
    config = config or ExperimentConfig()
    seed_everything(config.seed)
    device = torch.device(config.device)

    full_train = load_jsonl(config.train_path)
    test_examples = load_jsonl(config.test_path)
    train_examples, val_examples = make_train_val_split(
        full_train, val_size=config.val_size, seed=config.seed
    )

    tokenizer = TOKENIZERS[tokenizer_name]
    vocab, counter = build_vocab(
        [text for text, _ in train_examples], tokenizer, min_freq=config.min_freq
    )
    collate_fn = make_collate_fn(tokenizer, vocab)

    train_loader = DataLoader(
        train_examples, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_examples, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_examples, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = HumorClassifier(
        vocab_size=len(vocab),
        embed_dim=config.embed_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_examples = 0

        for text, offsets, labels in train_loader:
            text = text.to(device)
            offsets = offsets.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(text, offsets)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_train_loss += loss.item() * batch_size
            total_examples += batch_size

        val_metrics = evaluate_model(model, val_loader, loss_fn, device=str(device))
        epoch_result = {
            "epoch": epoch,
            "train_loss": total_train_loss / total_examples,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(epoch_result)

        if verbose:
            print(
                f"[{tokenizer_name}] epoch={epoch:02d} "
                f"train_loss={epoch_result['train_loss']:.4f} "
                f"val_accuracy={epoch_result['val_accuracy']:.4f} "
                f"val_f1={epoch_result['val_f1']:.4f}"
            )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model(model, test_loader, loss_fn, device=str(device))

    return {
        "tokenizer": tokenizer_name,
        "vocab_size": len(vocab),
        "observed_train_tokens": sum(counter.values()),
        "history": pd.DataFrame(history),
        "test_metrics": test_metrics,
        "model": model,
        "vocab": vocab,
        "train_examples": train_examples,
        "val_examples": val_examples,
        "test_examples": test_examples,
        "config": config,
    }


def run_all_experiments(
    config: ExperimentConfig | None = None,
    verbose: bool = True,
) -> tuple[dict[str, dict], pd.DataFrame]:
    config = config or ExperimentConfig()
    experiment_outputs = {}
    rows = []

    for tokenizer_name in TOKENIZERS:
        result = train_single_experiment(tokenizer_name, config=config, verbose=verbose)
        experiment_outputs[tokenizer_name] = result
        rows.append(
            {
                "tokenizer": tokenizer_name,
                "vocab_size": result["vocab_size"],
                **result["test_metrics"],
            }
        )

    results_df = pd.DataFrame(rows).sort_values(by=["f1", "accuracy"], ascending=False).reset_index(drop=True)
    return experiment_outputs, results_df

