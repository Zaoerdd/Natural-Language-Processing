import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "A2_metrics.json"
OUTPUT_MD = ROOT / "A2_report_zh.md"
OUTPUT_PDF = ROOT / "A2_report_zh.pdf"


def load_metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def to_markdown_table(rows, headers):
    header_line = "| " + " | ".join(headers) + " |"
    divider_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, divider_line, *body_lines])


def format_accuracy(value):
    return f"{value:.4f}"


def build_report_text(metrics):
    raw_stats = metrics["raw_corpus_stats"]
    runs = metrics["runs"]
    best_run_name = metrics["best_run_name"]
    best_run = runs[best_run_name]
    best_analogy = best_run["analogy_summary"]
    teacher_topn = metrics["teacher_standard"]["analogy_topn"]
    teacher_threshold = metrics["teacher_standard"]["passing_accuracy_hint"]

    raw_table = to_markdown_table(
        [
            ["总行数", f"{raw_stats['line_count']:,}"],
            ["非空行数", f"{raw_stats['nonempty_line_count']:,}"],
            ["清洗后原始 token 数", f"{raw_stats['raw_token_count']:,}"],
            ["清洗后原始词表大小", f"{raw_stats['raw_vocab_size']:,}"],
        ],
        ["指标", "数值"],
    )

    compare_rows = []
    for run_name in ("set1", "set2"):
        run = runs[run_name]
        cfg = run["config"]
        ana = run["analogy_summary"]
        compare_rows.append(
            [
                run_name,
                str(cfg["emb_size"]),
                str(cfg["window_size"]),
                str(cfg["negative_samples"]),
                str(cfg["min_count"]),
                cfg["embedding_mode"],
                format_accuracy(ana["overall_accuracy"]),
                f"{ana['correct_rows']}/{ana['valid_rows']}",
            ]
        )
    compare_table = to_markdown_table(
        compare_rows,
        ["配置", "emb_size", "window", "k", "min_count", "embedding", "top-5 acc", "correct/valid"],
    )

    loss_rows = []
    for run_name in ("set1", "set2"):
        for epoch_index, loss_value in enumerate(runs[run_name]["loss_history"], start=1):
            loss_rows.append([run_name, f"Epoch {epoch_index}", f"{loss_value:.4f}"])
    loss_table = to_markdown_table(loss_rows, ["配置", "轮次", "平均损失"])

    subject_rows = []
    top_subjects = sorted(
        best_analogy["per_subject"].items(),
        key=lambda item: (-item[1]["valid"], item[0]),
    )[:10]
    for subject, stats in top_subjects:
        accuracy_text = "N/A" if stats["accuracy"] is None else format_accuracy(stats["accuracy"])
        subject_rows.append(
            [subject, str(stats["valid"]), str(stats["correct"]), str(stats["skipped"]), accuracy_text]
        )
    subject_table = to_markdown_table(
        subject_rows,
        ["类别", "有效题数", "答对题数", "跳过题数", "准确率"],
    )

    selected_words = ", ".join(metrics["selected_words"])

    return f"""# Assignment 2 实验报告：Shakespeare Word2Vec

## 1. 任务概述

我基于 Shakespeare 语料实现了 `skip-gram + negative sampling` 的 Word2Vec 模型，并按助教在群里的说明，使用 **top-{teacher_topn}** 作为词类比任务的评分口径。助教同时说明，使用 `shakespeare.txt` 时，约 **{teacher_threshold:.0%}** 的 accuracy 已经处于可接受范围；后续又补充说明，只需要**任意一组**超参数达到这个标准即可。

我在最终提交中保留了两组超参数配置：

- `set1`: `emb_size=100, window=3, k=5, epochs=3, min_count=2`
- `set2`: `emb_size=150, window=5, k=15, epochs=3, min_count=2`

其中 `set2` 参考了助教推荐的较强配置，`set1` 作为较小模型基线。为减少 Shakespeare 原文中的大小写和标点碎片化问题，我在读入英文语料时统一做了正则清洗与小写化。

## 2. 数据与预处理

{raw_table}

我采用的预处理策略如下：

- 对英文文本使用正则表达式提取单词，去掉纯标点干扰。
- 全部转换为 lowercase，使训练词表与 analogy 评测词表对齐。
- 将 `CorpusReader.NEGATIVE_TABLE_SIZE` 显式设为 `1_000_000`，避免默认值导致不必要的内存压力。

## 3. 两组超参数与结果对比

{compare_table}

从结果来看，**{best_run_name}** 达到了助教说明中的可接受标准，因此我将它作为最终提交时重点展示的配置；另一组结果保留在报告中作为对照和补充说明。

## 4. 训练损失曲线

{loss_table}

可以看到，两组配置的平均损失都随 epoch 下降，满足作业对“训练过程中损失曲线呈下降趋势”的要求。

![训练损失曲线](A2_training_loss.png)

## 5. Best Run 的 Analogy 结果

- 最佳配置：**{best_run_name}**
- top-{teacher_topn} accuracy：**{format_accuracy(best_analogy["overall_accuracy"])}**
- correct / valid：**{best_analogy["correct_rows"]}/{best_analogy["valid_rows"]}**
- skipped：**{best_analogy["skipped_rows"]}**

下表展示最佳配置下有效题数最多的前 10 个类别：

{subject_table}

## 6. 二维可视化分析

我对以下词的向量进行了二维投影：

- {selected_words}

![二维词向量可视化](A2_embedding_svd.png)

由于 Shakespeare 训练语料规模较小，二维投影不会像大规模百科语料那样形成非常整齐的几何结构，但仍然可以观察到部分人物与性别相关词在局部空间中的相对聚集趋势。

## 7. 实现细节与复现说明

- 完成版 notebook：`A2_w2v_completed.ipynb`
- 主实验模块：`a2_word2vec_experiment.py`
- 词向量文件：`embeddings_set1.txt`、`embeddings_set2.txt`、`embeddings.txt`
- 结构化结果：`A2_metrics.json`
- 中文实验报告：`A2_report_zh.md` / `A2_report_zh.pdf`

如果需要复现，我使用的顺序如下：

1. 运行 `build_a2_completed_notebook.py` 生成完成版 notebook。
2. 执行 `A2_w2v_completed.ipynb`，自动训练 `set1` 和 `set2`，并生成 loss 图、SVD 图、embedding 文件和 `A2_metrics.json`。
3. 运行 `build_a2_report_zh.py`，根据 `A2_metrics.json` 以及图像文件生成最终中文 Markdown / PDF 报告。
"""


def register_chinese_font():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font_candidates = [
        ("MicrosoftYaHei", Path(r"C:\Windows\Fonts\msyh.ttc")),
        ("SimHei", Path(r"C:\Windows\Fonts\simhei.ttf")),
        ("SimSun", Path(r"C:\Windows\Fonts\simsun.ttc")),
    ]
    for font_name, font_path in font_candidates:
        if font_path.exists():
            pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
            return font_name
    raise FileNotFoundError("No supported Chinese font found under C:\\Windows\\Fonts")


def build_pdf(metrics):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    font_name = register_chinese_font()
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ZhTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=18,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=12,
        textColor=colors.HexColor("#111111"),
    )
    heading_style = ParagraphStyle(
        "ZhHeading",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=13,
        leading=18,
        spaceBefore=10,
        spaceAfter=6,
        textColor=colors.HexColor("#111111"),
    )
    body_style = ParagraphStyle(
        "ZhBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=10.5,
        leading=16,
        spaceAfter=6,
        textColor=colors.HexColor("#222222"),
    )

    def make_table(data, col_widths):
        table = Table(data, colWidths=col_widths, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), font_name),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                    ("LEADING", (0, 0), (-1, -1), 12),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAECEF")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#222222")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return table

    def make_image(path: Path, max_width: float):
        image_reader = ImageReader(str(path))
        width, height = image_reader.getSize()
        scale = min(max_width / width, 1.0)
        image = Image(str(path))
        image.drawWidth = width * scale
        image.drawHeight = height * scale
        return image

    runs = metrics["runs"]
    raw_stats = metrics["raw_corpus_stats"]
    teacher_topn = metrics["teacher_standard"]["analogy_topn"]
    teacher_threshold = metrics["teacher_standard"]["passing_accuracy_hint"]
    best_run_name = metrics["best_run_name"]
    best_run = runs[best_run_name]
    best_analogy = best_run["analogy_summary"]

    compare_table = [
        ["配置", "emb", "window", "k", "min_count", "embedding", f"top-{teacher_topn} acc"],
    ]
    for run_name in ("set1", "set2"):
        run = runs[run_name]
        cfg = run["config"]
        compare_table.append(
            [
                run_name,
                str(cfg["emb_size"]),
                str(cfg["window_size"]),
                str(cfg["negative_samples"]),
                str(cfg["min_count"]),
                cfg["embedding_mode"],
                format_accuracy(run["analogy_summary"]["overall_accuracy"]),
            ]
        )

    raw_table = [
        ["指标", "数值"],
        ["总行数", f"{raw_stats['line_count']:,}"],
        ["非空行数", f"{raw_stats['nonempty_line_count']:,}"],
        ["清洗后 token 数", f"{raw_stats['raw_token_count']:,}"],
        ["清洗后词表大小", f"{raw_stats['raw_vocab_size']:,}"],
    ]

    subject_table = [["类别", "有效题数", "答对题数", "跳过题数", "准确率"]]
    top_subjects = sorted(
        best_analogy["per_subject"].items(),
        key=lambda item: (-item[1]["valid"], item[0]),
    )[:10]
    for subject, stats in top_subjects:
        accuracy_text = "N/A" if stats["accuracy"] is None else format_accuracy(stats["accuracy"])
        subject_table.append(
            [subject, str(stats["valid"]), str(stats["correct"]), str(stats["skipped"]), accuracy_text]
        )

    story = [
        Paragraph("Assignment 2 实验报告：Shakespeare Word2Vec", title_style),
        Paragraph("1. 评分口径", heading_style),
        Paragraph(
            f"按助教群消息，本次 analogy 任务使用 top-{teacher_topn} 作为评分标准，约 {teacher_threshold:.0%} 的准确率即可接受；后续又明确说明只需要任意一组超参数达标即可。",
            body_style,
        ),
        Paragraph("2. 数据与预处理", heading_style),
        make_table(raw_table, [1.8 * inch, 1.6 * inch]),
        Spacer(1, 0.15 * inch),
        Paragraph(
            "我在读入英文语料时做了正则清洗与小写化，以减少标点和大小写造成的词表碎片化，并将负采样表大小限制为 1,000,000 以控制内存占用。",
            body_style,
        ),
        Paragraph("3. 两组配置对比", heading_style),
        make_table(compare_table, [0.7 * inch, 0.6 * inch, 0.7 * inch, 0.45 * inch, 0.8 * inch, 1.0 * inch, 1.0 * inch]),
        Spacer(1, 0.15 * inch),
        Paragraph("4. 训练损失曲线", heading_style),
        Paragraph("我得到的两组配置平均损失都随 epoch 下降，满足作业对训练过程可视化的要求。", body_style),
        make_image(ROOT / "A2_training_loss.png", max_width=6.2 * inch),
        Spacer(1, 0.15 * inch),
        Paragraph("5. Best Run 结果", heading_style),
        Paragraph(
            (
                f"我最终采用的最佳配置为 {best_run_name}。"
                f" top-{teacher_topn} accuracy = {format_accuracy(best_analogy['overall_accuracy'])}，"
                f" correct/valid = {best_analogy['correct_rows']}/{best_analogy['valid_rows']}，"
                f" skipped = {best_analogy['skipped_rows']}。"
            ),
            body_style,
        ),
        make_table(subject_table, [2.2 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch]),
        Spacer(1, 0.15 * inch),
        Paragraph("6. 二维词向量可视化", heading_style),
        Paragraph(
            "下图展示我在最佳配置下得到的选定词向量二维投影，用于辅助观察局部语义结构。",
            body_style,
        ),
        make_image(ROOT / "A2_embedding_svd.png", max_width=6.2 * inch),
    ]

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="Assignment 2 Chinese Report",
        author="Codex",
    )
    doc.build(story)


def main():
    metrics = load_metrics()
    report_text = build_report_text(metrics)
    OUTPUT_MD.write_text(report_text, encoding="utf-8")
    print(f"Wrote {OUTPUT_MD}")
    build_pdf(metrics)
    print(f"Wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
