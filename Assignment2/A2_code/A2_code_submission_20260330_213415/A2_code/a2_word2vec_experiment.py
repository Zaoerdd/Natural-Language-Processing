import csv
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors
from sklearn.decomposition import TruncatedSVD

from utils import CorpusReader, tokenize_line


SEED = 42
DEVICE = torch.device("cpu")
NEGATIVE_TABLE_SIZE = 1_000_000
ANALOGY_TOPN = 5


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_raw_stats(input_file: str):
    token_counter = Counter()
    line_count = 0
    nonempty_line_count = 0
    for raw_line in Path(input_file).read_text(encoding="utf-8").splitlines():
        line_count += 1
        tokens = tokenize_line(raw_line, "en")
        if tokens:
            nonempty_line_count += 1
            token_counter.update(tokens)
    return {
        "line_count": line_count,
        "nonempty_line_count": nonempty_line_count,
        "raw_token_count": int(sum(token_counter.values())),
        "raw_vocab_size": int(len(token_counter)),
    }, token_counter


def iter_corpus_lines(input_file: str, corpus: CorpusReader):
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            words = [word for word in tokenize_line(line, "en") if word in corpus.word2id]
            if len(words) >= 2:
                yield words


def generate_data(words, window_size: int, k: int, corpus: CorpusReader):
    word_ids = [corpus.word2id[word] for word in words if word in corpus.word2id]
    for center_pos, center_id in enumerate(word_ids):
        left = max(0, center_pos - window_size)
        right = min(len(word_ids), center_pos + window_size + 1)
        for outside_pos in range(left, right):
            if outside_pos == center_pos:
                continue
            outside_id = word_ids[outside_pos]
            negative_ids = corpus.getNegatives(outside_id, k).astype(np.int64, copy=False)
            yield center_id, outside_id, negative_ids


def batchify(data_stream, batch_size: int):
    batch = []
    for sample in data_stream:
        batch.append(sample)
        if len(batch) == batch_size:
            yield build_batch_tensors(batch)
            batch = []
    if batch:
        yield build_batch_tensors(batch)


def build_batch_tensors(batch):
    center = torch.tensor([item[0] for item in batch], dtype=torch.long)
    outside = torch.tensor([item[1] for item in batch], dtype=torch.long)
    negative = torch.tensor(np.stack([item[2] for item in batch]), dtype=torch.long)
    return center, outside, negative


def count_training_pairs(input_file: str, window_size: int, corpus: CorpusReader):
    usable_lines = 0
    total_pairs = 0
    for words in iter_corpus_lines(input_file, corpus):
        usable_lines += 1
        for center_pos in range(len(words)):
            left = max(0, center_pos - window_size)
            right = min(len(words), center_pos + window_size + 1)
            total_pairs += right - left - 1
    return int(usable_lines), int(total_pairs)


class SkipGram(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.emb_v = nn.Embedding(vocab_size, emb_size, sparse=True)
        self.emb_u = nn.Embedding(vocab_size, emb_size, sparse=True)

        initrange = 1.0 / emb_size
        nn.init.uniform_(self.emb_v.weight.data, -initrange, initrange)
        nn.init.constant_(self.emb_u.weight.data, 0)

    def forward(self, center, outside, negative):
        v_c = self.emb_v(center)
        u_o = self.emb_u(outside)
        u_n = self.emb_u(negative)

        positive_score = torch.sum(v_c * u_o, dim=1).clamp(-10, 10)
        positive_loss = F.logsigmoid(positive_score)

        negative_score = torch.bmm(u_n, v_c.unsqueeze(2)).squeeze(2).clamp(-10, 10)
        negative_loss = F.logsigmoid(-negative_score).sum(dim=1)
        return -(positive_loss + negative_loss)


def prepare_corpus(input_file: str, min_count: int):
    CorpusReader.NEGATIVE_TABLE_SIZE = NEGATIVE_TABLE_SIZE
    return CorpusReader(input_file, min_count=min_count, lang="en")


def save_embedding(model: SkipGram, corpus: CorpusReader, file_name: str, mode: str):
    input_weights = model.emb_v.weight.detach().cpu().numpy()
    output_weights = model.emb_u.weight.detach().cpu().numpy()
    if mode == "input":
        weights = input_weights
    elif mode == "combined":
        weights = (input_weights + output_weights) / 2.0
    else:
        raise ValueError(f"Unsupported embedding mode: {mode}")

    with open(file_name, "w", encoding="utf-8") as f:
        f.write(f"{len(corpus.id2word)} {weights.shape[1]}\n")
        for wid, word in corpus.id2word.items():
            f.write(word + " " + " ".join(map(str, weights[wid])) + "\n")


def evaluate_analogies(embedding_path: str, csv_path: str, topn: int = ANALOGY_TOPN):
    wv = KeyedVectors.load_word2vec_format(embedding_path, binary=False)
    per_subject = {}
    total_rows = 0
    valid_rows = 0
    skipped_rows = 0
    correct_rows = 0

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            w1, w2, w3, w4 = [row[key].lower() for key in ("Word1", "Word2", "Word3", "Word4")]
            subject = row["Subject"]

            stats = per_subject.setdefault(
                subject,
                {"correct": 0, "valid": 0, "skipped": 0, "examples": []},
            )

            if any(word not in wv for word in (w1, w2, w3, w4)):
                skipped_rows += 1
                stats["skipped"] += 1
                continue

            candidates = [
                word
                for word, _score in wv.most_similar(positive=[w2, w3], negative=[w1], topn=50)
                if word not in {w1, w2, w3}
            ][:topn]

            if not candidates:
                skipped_rows += 1
                stats["skipped"] += 1
                continue

            is_correct = w4 in candidates
            valid_rows += 1
            correct_rows += int(is_correct)
            stats["valid"] += 1
            stats["correct"] += int(is_correct)

            if len(stats["examples"]) < 3:
                stats["examples"].append(
                    {
                        "prompt": f"{w1}:{w2}::{w3}:?",
                        "predictions": candidates,
                        "target": w4,
                        "correct": bool(is_correct),
                    }
                )

    for stats in per_subject.values():
        stats["accuracy"] = float(stats["correct"] / stats["valid"]) if stats["valid"] else None

    return {
        "topn": int(topn),
        "total_rows": int(total_rows),
        "valid_rows": int(valid_rows),
        "skipped_rows": int(skipped_rows),
        "correct_rows": int(correct_rows),
        "overall_accuracy": float(correct_rows / valid_rows) if valid_rows else 0.0,
        "per_subject": per_subject,
    }


def train_run(input_file: str, analogy_file: str, output_dir: str, run_config: dict):
    corpus = prepare_corpus(input_file, min_count=run_config["min_count"])
    usable_lines, training_pairs = count_training_pairs(input_file, run_config["window_size"], corpus)

    model = SkipGram(corpus.vocab_size, run_config["emb_size"]).to(DEVICE)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=run_config["learning_rate"])

    loss_history = []
    for epoch in range(run_config["epochs"]):
        total_loss = 0.0
        total_examples = 0
        sentences = list(iter_corpus_lines(input_file, corpus))
        random.shuffle(sentences)
        for center, outside, negative in batchify(
            (
                sample
                for words in sentences
                for sample in generate_data(words, run_config["window_size"], run_config["negative_samples"], corpus)
            ),
            batch_size=run_config["batch_size"],
        ):
            center = center.to(DEVICE)
            outside = outside.to(DEVICE)
            negative = negative.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            loss = model(center, outside, negative).mean()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * center.size(0)
            total_examples += center.size(0)

        loss_history.append(total_loss / total_examples)

    embedding_file = str(Path(output_dir) / f"embeddings_{run_config['name']}.txt")
    save_embedding(model, corpus, embedding_file, run_config["embedding_mode"])
    analogy_summary = evaluate_analogies(embedding_file, analogy_file, topn=run_config["analogy_topn"])

    return {
        "name": run_config["name"],
        "config": run_config,
        "corpus_stats": {
            "filtered_token_count": int(sum(corpus.word_frequency.values())),
            "filtered_vocab_size": int(corpus.vocab_size),
            "usable_lines": usable_lines,
            "training_pairs": training_pairs,
        },
        "loss_history": [float(value) for value in loss_history],
        "embedding_file": str(Path(embedding_file).resolve()),
        "analogy_summary": analogy_summary,
    }


def plot_loss_histories(results: list, output_file: str):
    plt.figure(figsize=(7, 4.5))
    for result in results:
        epochs = range(1, len(result["loss_history"]) + 1)
        label = f"{result['name']} ({result['config']['embedding_mode']})"
        plt.plot(epochs, result["loss_history"], marker="o", label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss by Hyper-parameter Set")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


def project_embeddings(embedding_path: str, selected_words: list, output_file: str):
    wv = KeyedVectors.load_word2vec_format(embedding_path, binary=False)
    available_words = [word for word in selected_words if word in wv]
    vectors = np.vstack([wv[word] for word in available_words])
    svd = TruncatedSVD(n_components=2, random_state=42)
    vectors_2d = svd.fit_transform(vectors)

    plt.figure(figsize=(8, 6))
    for index, word in enumerate(available_words):
        plt.scatter(vectors_2d[index, 0], vectors_2d[index, 1], s=60)
        plt.text(vectors_2d[index, 0] + 0.01, vectors_2d[index, 1] + 0.01, word)
    plt.title("2D projection of selected Shakespeare embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "selected_words": available_words,
        "explained_variance_ratio": float(svd.explained_variance_ratio_.sum()),
    }


def run_experiments(input_file: str, analogy_file: str, output_dir: str):
    set_seed(SEED)
    output_dir_path = Path(output_dir)

    raw_stats, _ = collect_raw_stats(input_file)
    common_config = {
        "seed": SEED,
        "epochs": 3,
        "batch_size": 128,
        "learning_rate": 0.025,
        "negative_table_size": NEGATIVE_TABLE_SIZE,
        "analogy_topn": ANALOGY_TOPN,
        "device": str(DEVICE),
    }
    run_configs = [
        {
            "name": "set1",
            "emb_size": 100,
            "window_size": 3,
            "negative_samples": 5,
            "min_count": 2,
            "embedding_mode": "input",
            **common_config,
        },
        {
            "name": "set2",
            "emb_size": 150,
            "window_size": 5,
            "negative_samples": 15,
            "min_count": 2,
            "embedding_mode": "combined",
            **common_config,
        },
    ]

    results = [train_run(input_file, analogy_file, output_dir, config) for config in run_configs]
    best_result = max(
        results,
        key=lambda item: (
            item["analogy_summary"]["overall_accuracy"],
            item["analogy_summary"]["valid_rows"],
        ),
    )

    best_alias = output_dir_path / "embeddings.txt"
    Path(best_alias).write_text(Path(best_result["embedding_file"]).read_text(encoding="utf-8"), encoding="utf-8")

    loss_plot_file = str((output_dir_path / "A2_training_loss.png").resolve())
    plot_loss_histories(results, loss_plot_file)

    selected_words = ["sister", "brother", "woman", "man", "girl", "boy", "queen", "king"]
    svd_plot_file = str((output_dir_path / "A2_embedding_svd.png").resolve())
    svd_summary = project_embeddings(best_result["embedding_file"], selected_words, svd_plot_file)

    metrics = {
        "teacher_standard": {
            "analogy_topn": ANALOGY_TOPN,
            "passing_accuracy_hint": 0.01,
        },
        "raw_corpus_stats": raw_stats,
        "runs": {result["name"]: result for result in results},
        "best_run_name": best_result["name"],
        "selected_words": svd_summary["selected_words"],
        "svd_summary": svd_summary,
        "artifact_paths": {
            "loss_plot": loss_plot_file,
            "svd_plot": svd_plot_file,
            "best_embedding_file": str(best_alias.resolve()),
            "completed_notebook": str((output_dir_path / "A2_w2v_completed.ipynb").resolve()),
        },
    }

    metrics_path = output_dir_path / "A2_metrics.json"
    metrics["artifact_paths"]["metrics_file"] = str(metrics_path.resolve())
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics
