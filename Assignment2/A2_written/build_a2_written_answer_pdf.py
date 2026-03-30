from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "A2_w2v_written_ans.md"
OUTPUT = ROOT / "A2_w2v_written_ans.pdf"


def parse_blocks(text: str):
    blocks = []
    lines = text.splitlines()
    paragraph = []
    code = []
    in_code = False

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            blocks.append(("paragraph", " ".join(line.strip() for line in paragraph)))
            paragraph = []

    for line in lines:
        stripped = line.rstrip()

        if stripped.startswith("```"):
            if in_code:
                blocks.append(("code", "\n".join(code).strip("\n")))
                code = []
                in_code = False
            else:
                flush_paragraph()
                in_code = True
            continue

        if in_code:
            code.append(stripped)
            continue

        if not stripped:
            flush_paragraph()
            continue

        if stripped.startswith("# "):
            flush_paragraph()
            blocks.append(("title", stripped[2:].strip()))
            continue

        if stripped.startswith("## "):
            flush_paragraph()
            blocks.append(("heading", stripped[3:].strip()))
            continue

        paragraph.append(stripped)

    flush_paragraph()
    return blocks


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawRightString(doc.pagesize[0] - 0.7 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf():
    text = SOURCE.read_text(encoding="utf-8")
    blocks = parse_blocks(text)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#111111"),
        spaceAfter=10,
    )
    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#111111"),
        spaceBefore=8,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#222222"),
        spaceAfter=6,
    )
    code_style = ParagraphStyle(
        "CodeCustom",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=9.5,
        leading=12,
        leftIndent=14,
        rightIndent=8,
        borderWidth=0.5,
        borderColor=colors.HexColor("#CCCCCC"),
        borderPadding=8,
        backColor=colors.HexColor("#F7F7F7"),
        textColor=colors.HexColor("#111111"),
        spaceAfter=8,
    )

    story = []
    for kind, content in blocks:
        if kind == "title":
            story.append(Paragraph(escape(content), title_style))
            story.append(Spacer(1, 0.04 * inch))
        elif kind == "heading":
            story.append(Paragraph(escape(content), heading_style))
        elif kind == "paragraph":
            story.append(Paragraph(escape(content), body_style))
        elif kind == "code":
            story.append(Preformatted(content, code_style))

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.8 * inch,
        title="A2 word2vec written answers",
        author="Codex",
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


if __name__ == "__main__":
    build_pdf()
