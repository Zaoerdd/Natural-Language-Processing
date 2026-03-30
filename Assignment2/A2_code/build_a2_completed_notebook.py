from pathlib import Path
import textwrap

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "A2_w2v.ipynb"
OUTPUT = ROOT / "A2_w2v_completed.ipynb"


def code(source: str):
    return textwrap.dedent(source).strip("\n") + "\n"


def markdown(source: str):
    return textwrap.dedent(source).strip("\n")


IMPORTS_CELL = code(
    """
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import torch
    from gensim.models import KeyedVectors

    from a2_word2vec_experiment import (
        ANALOGY_TOPN,
        DEVICE,
        NEGATIVE_TABLE_SIZE,
        SEED,
        SkipGram,
        batchify,
        collect_raw_stats,
        generate_data,
        iter_corpus_lines,
        prepare_corpus,
        run_experiments,
    )
    """
)


DATA_SETUP_CELL = code(
    """
    INPUT_FILE = "shakespeare.txt"
    ANALOGY_FILE = "questions-words-shakespeare.csv"
    OUTPUT_DIR = "."

    raw_corpus_stats, token_counter = collect_raw_stats(INPUT_FILE)
    print("Teacher analogy standard: topn =", ANALOGY_TOPN)
    print("Negative table size:", NEGATIVE_TABLE_SIZE)
    print("Device:", DEVICE)
    print(json.dumps(raw_corpus_stats, indent=2))

    preview_corpus = prepare_corpus(INPUT_FILE, min_count=2)
    print(f"Preview corpus vocab size (min_count=2): {preview_corpus.vocab_size:,}")
    """
)


DATA_FUNCTIONS_CELL = code(
    """
    sample_words = next(iter(iter_corpus_lines(INPUT_FILE, preview_corpus)))[:6]
    sample_data = list(generate_data(sample_words, window_size=3, k=5, corpus=preview_corpus))
    sample_batches = list(batchify(sample_data, batch_size=4))

    print("Sample words:", sample_words)
    print("Sample pair count:", len(sample_data))
    print("First sample:", sample_data[0])
    print("First batch shapes:", sample_batches[0][0].shape, sample_batches[0][1].shape, sample_batches[0][2].shape)

    assert len(sample_data) > 0
    assert sample_batches[0][2].shape[1] == 5
    """
)


MODEL_CELL = code(
    """
    smoke_model = SkipGram(preview_corpus.vocab_size, emb_size=32)
    center_smoke, outside_smoke, negative_smoke = sample_batches[0]

    with torch.no_grad():
        smoke_loss = smoke_model(center_smoke, outside_smoke, negative_smoke)

    print("Smoke loss shape:", smoke_loss.shape)
    print("Smoke loss mean:", float(smoke_loss.mean()))
    assert smoke_loss.shape[0] == center_smoke.shape[0]
    assert torch.isfinite(smoke_loss).all()
    """
)


TRAIN_CELL = code(
    """
    metrics = run_experiments(INPUT_FILE, ANALOGY_FILE, OUTPUT_DIR)
    print("Best run:", metrics["best_run_name"])
    print(json.dumps(metrics["teacher_standard"], indent=2))
    metrics
    """
)


SAVE_EMBEDDING_CELL = code(
    """
    for run_name, run_result in metrics["runs"].items():
        print(run_name, "embedding:", run_result["embedding_file"])
        print(run_name, "top-5 accuracy:", f"{run_result['analogy_summary']['overall_accuracy']:.4f}")

    print("Best embedding alias:", metrics["artifact_paths"]["best_embedding_file"])
    """
)


ANALOGY_MARKDOWN = markdown(
    """
    ## 4.5 Analogy Evaluation

    The grading standard is `topn = 5`. A prediction is counted as correct if the target word
    appears in the top-5 analogy candidates after excluding the query words themselves.
    """
)


ANALOGY_CELL = code(
    """
    for run_name, run_result in metrics["runs"].items():
        summary = run_result["analogy_summary"]
        print(
            f"{run_name}: top-{summary['topn']} accuracy = {summary['overall_accuracy']:.4f} "
            f"({summary['correct_rows']}/{summary['valid_rows']}), skipped = {summary['skipped_rows']}"
        )

    best_run = metrics["runs"][metrics["best_run_name"]]
    top_subjects = sorted(
        best_run["analogy_summary"]["per_subject"].items(),
        key=lambda item: (-item[1]["valid"], item[0]),
    )[:10]

    for subject, stats in top_subjects:
        accuracy = stats["accuracy"]
        accuracy_text = "N/A" if accuracy is None else f"{accuracy:.4f}"
        print(
            f"{subject:30s} valid={stats['valid']:4d} skipped={stats['skipped']:4d} accuracy={accuracy_text}"
        )
    """
)


LOAD_EMBEDDINGS_CELL = code(
    """
    best_embedding_file = metrics["artifact_paths"]["best_embedding_file"]
    wv_from_file = KeyedVectors.load_word2vec_format(best_embedding_file, binary=False)

    embedding_words = list(wv_from_file.index_to_key)
    embedding_matrix = wv_from_file.vectors.copy()

    print("Loaded best embedding file:", best_embedding_file)
    print("Embedding matrix shape:", embedding_matrix.shape)
    """
)


SVD_CELL = code(
    """
    selected_words = metrics["selected_words"]
    vectors_2d = []
    for word in selected_words:
        vectors_2d.append(wv_from_file[word])

    import numpy as np
    from sklearn.decomposition import TruncatedSVD

    vectors_2d = np.vstack(vectors_2d)
    svd = TruncatedSVD(n_components=2, random_state=42)
    vectors_2d = svd.fit_transform(vectors_2d)

    print("Selected words:", selected_words)
    print("vectors_2d shape:", vectors_2d.shape)
    print("Explained variance ratio:", float(svd.explained_variance_ratio_.sum()))
    """
)


PLOT_CELL = code(
    """
    words = selected_words

    plt.figure(figsize=(8, 6))
    for index, word in enumerate(words):
        plt.scatter(vectors_2d[index, 0], vectors_2d[index, 1], s=60)
        plt.text(vectors_2d[index, 0] + 0.01, vectors_2d[index, 1] + 0.01, word)

    plt.title("2D projection of selected Shakespeare embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Saved plot file:", metrics["artifact_paths"]["svd_plot"])
    """
)


METRICS_MARKDOWN = markdown(
    """
    ## 6. Save Structured Metrics

    Metrics are written by `run_experiments(...)` and reused by the final Chinese report builder.
    """
)


METRICS_CELL = code(
    """
    metrics_path = Path(metrics["artifact_paths"]["metrics_file"])
    print("Metrics file:", metrics_path.resolve())
    print(metrics_path.read_text(encoding="utf-8")[:1200])
    """
)


def build_notebook():
    notebook = nbformat.read(TEMPLATE, as_version=4)
    notebook.nbformat = 4
    notebook.nbformat_minor = 5

    notebook.cells[2].source = IMPORTS_CELL
    notebook.cells[4].source = DATA_SETUP_CELL
    notebook.cells[5].source = DATA_FUNCTIONS_CELL
    notebook.cells[7].source = MODEL_CELL
    notebook.cells[9].source = TRAIN_CELL
    notebook.cells[11].source = SAVE_EMBEDDING_CELL
    notebook.cells[13].source = LOAD_EMBEDDINGS_CELL
    notebook.cells[14].source = SVD_CELL
    notebook.cells[15].source = PLOT_CELL

    final_cells = []
    for index, cell in enumerate(notebook.cells):
        final_cells.append(cell)
        if index == 11:
            final_cells.append(new_markdown_cell(ANALOGY_MARKDOWN))
            final_cells.append(new_code_cell(ANALOGY_CELL))
        elif index == 15:
            final_cells.append(new_markdown_cell(METRICS_MARKDOWN))
            final_cells.append(new_code_cell(METRICS_CELL))

    notebook.cells = final_cells
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"a2-cell-{index:03d}"
    nbformat.write(notebook, OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build_notebook()
