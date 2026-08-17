#!/usr/bin/env python
"""Convert AquaContam paper markdown to Word (.docx) format.

Two conversion strategies:
1. **Primary (pandoc)**: Preprocess markdown, run pandoc with reference
   template, post-process with python-docx.
2. **Fallback (pure python-docx)**: Build document programmatically.

Usage::

    python paper/convert_to_docx.py
    python paper/convert_to_docx.py --extended-data --supplementary
    python paper/convert_to_docx.py --all
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

PAPER_DIR = Path(__file__).resolve().parent
DEFAULT_SKELETON = PAPER_DIR / "skeleton.md"
DEFAULT_OUTPUT = PAPER_DIR / "manuscript.docx"

# Nature Water approximate styling
FONT_NAME = "Arial"
FONT_SIZE_PT = 11
LINE_SPACING = 1.5  # default for the individual submission docs
REVIEW_LINE_SPACING = 1.0  # true single spacing for the readable review copy
MARGIN_INCHES = 1.0

# Table rendering (shared by pandoc post-process and pure python-docx fallback)
TABLE_FONT_SIZE_PT = 9  # default table cell font
WIDE_TABLE_COL_THRESHOLD = 12  # tables wider than this shrink further
WIDE_TABLE_FONT_SIZE_PT = 6  # cell font for very wide tables (e.g. 44-col per-analyte)

# Figure sizing: US Letter (8.5in) minus 2x1in margins => 6.5in usable text width.
MAX_IMAGE_WIDTH_IN = 6.5

_MAIN_FIGURE_FILES: dict[int, str] = {
    1: "figures/fig1_dataset_overview.png",
    2: "figures/fig2_confound_decomposition.png",
    3: "figures/fig3_shap_summary.png",
    4: "figures/fig4_monitoring_inequity.png",
}
_MAIN_FIGURE_CAPTIONS: dict[int, str] = {
    1: "Fig. 1: National-scale data integration",
    2: "Fig. 2: Two compounding confounds inflate reported skill",
    3: "Fig. 3: Models learn the monitoring process (SHAP)",
    4: "Fig. 4: Inequitable monitoring",
}
_MAIN_TABLE_FILES: dict[int, str] = {
    1: "tables/table1_dataset_summary.md",
    2: "tables/table2_benchmark_condensed.md",
    3: "tables/table3_monitoring_invariance.md",
}


def _has_pandoc() -> bool:
    """Check whether pandoc is available on PATH."""
    return shutil.which("pandoc") is not None


def _preprocess_markdown(md_path: Path) -> str:
    """Preprocess markdown for pandoc conversion.

    Resolves relative figure paths to absolute paths and ensures
    tables are embedded inline.

    Parameters
    ----------
    md_path : Path
        Path to the markdown file.

    Returns
    -------
    str
        Preprocessed markdown content.
    """
    text = md_path.read_text(encoding="utf-8")
    text = _resolve_figure_paths(text)

    # Strip Extended Data section from main manuscript only (not from extended_data.md)
    if md_path.name != "extended_data.md":
        ed_marker = "### Extended Data Fig."
        if ed_marker in text:
            idx = text.index(ed_marker)
            # Find the last heading before ED section
            text = text[:idx].rstrip() + "\n"

    return text


def _image_width_attr(img_path: Path) -> str:
    """Return a pandoc image width attribute clamped to the text width.

    Reads the PNG's pixel width and DPI; the rendered width is
    ``min(native_inches, MAX_IMAGE_WIDTH_IN)`` so oversized figures clamp to the
    page and smaller figures are never upscaled. Robust to missing/zero DPI
    metadata (falls back to 96 dpi).
    """
    try:
        from PIL import Image

        with Image.open(img_path) as im:
            px_width = im.size[0]
            dpi = im.info.get("dpi", (96, 96))
            x_dpi = dpi[0] if dpi and dpi[0] else 96
        native_in = px_width / float(x_dpi)
        width_in = min(native_in, MAX_IMAGE_WIDTH_IN)
    except Exception:
        # If the image can't be read, clamp to the safe maximum.
        width_in = MAX_IMAGE_WIDTH_IN
    return f"{{width={round(width_in, 2)}in}}"


def _resolve_figure_paths(text: str) -> str:
    """Resolve relative figure paths to absolute paths and clamp their width.

    Parameters
    ----------
    text : str
        Markdown content with relative image paths.

    Returns
    -------
    str
        Markdown with absolute image paths and a ``{width=..in}`` attribute that
        keeps every figure within the page text width under pandoc.
    """
    fig_dir = PAPER_DIR / "figures"
    if not fig_dir.exists():
        return text

    def _resolve_figure(match: re.Match[str]) -> str:
        alt = match.group(1)
        rel_path = match.group(2)
        candidate = PAPER_DIR / rel_path
        if candidate.exists():
            return f"![{alt}]({candidate}){_image_width_attr(candidate)}"
        png = candidate.with_suffix(".png")
        if png.exists():
            return f"![{alt}]({png}){_image_width_attr(png)}"
        return match.group(0)

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _resolve_figure, text)


def _find_main_refs_in_paragraph(paragraph: str) -> tuple[set[int], set[int]]:
    """Find main figure and table references in a paragraph.

    Strips Extended Data and Supplementary references first to avoid
    false matches (e.g. "Extended Data Fig. 4" should not match "Fig. 4").

    Parameters
    ----------
    paragraph : str
        A single paragraph of markdown text.

    Returns
    -------
    tuple[set[int], set[int]]
        Sets of (figure_numbers, table_numbers) referenced in the paragraph.
    """
    # Collapse internal whitespace first so a qualifier that the source hard-wraps
    # ("…Extended Data\nTable 3") still matches the literal-space strip below;
    # otherwise the bare "Table 3" leaks and is injected in the wrong place.
    paragraph = re.sub(r"\s+", " ", paragraph)
    # Remove qualified references so only bare "Fig. N" / "Table N" remain
    stripped = re.sub(r"Extended Data (?:Fig\.|Table)\s*\d+", "", paragraph)
    stripped = re.sub(r"Supplementary (?:Fig\.|Table)\s*\d+", "", stripped)

    fig_nums = {int(m) for m in re.findall(r"Fig\.\s*(\d+)", stripped)} & _MAIN_FIGURE_FILES.keys()
    tbl_nums = {int(m) for m in re.findall(r"Table\s+(\d+)", stripped)} & _MAIN_TABLE_FILES.keys()

    return fig_nums, tbl_nums


def _strip_leading_headings(content: str) -> str:
    """Drop leading markdown heading and blank lines from a table file's content.

    Table files such as ``table3_monitoring_invariance.md`` begin with a
    ``## Table 3: …`` heading that would otherwise render as a rogue document
    heading when the table is injected inline; the injected ``**Table N**`` label
    already supplies the caption.
    """
    lines = content.split("\n")
    while lines and (lines[0].startswith("#") or lines[0].strip() == ""):
        lines.pop(0)
    return "\n".join(lines)


def _insert_main_figures_and_tables(text: str) -> str:
    """Insert main figures and tables after the paragraph where first mentioned.

    Parameters
    ----------
    text : str
        Raw skeleton markdown content.

    Returns
    -------
    str
        Markdown with figure images and table content injected inline.
    """
    paragraphs = re.split(r"\n\n+", text)
    result: list[str] = []
    inserted_figs: set[int] = set()
    inserted_tbls: set[int] = set()

    for para in paragraphs:
        result.append(para)

        fig_refs, tbl_refs = _find_main_refs_in_paragraph(para)

        # Insert figures (sorted by number for deterministic order)
        for n in sorted(fig_refs - inserted_figs):
            caption = _MAIN_FIGURE_CAPTIONS.get(n, f"Fig. {n}")
            rel_path = _MAIN_FIGURE_FILES[n]
            fig_path = PAPER_DIR / rel_path
            if fig_path.exists():
                result.append(f"![{caption}]({rel_path})")
            else:
                logger.warning("Figure file not found: %s", fig_path)
            inserted_figs.add(n)

        # Insert tables (sorted by number)
        for n in sorted(tbl_refs - inserted_tbls):
            rel_path = _MAIN_TABLE_FILES[n]
            tbl_path = PAPER_DIR / rel_path
            if tbl_path.exists():
                tbl_content = _strip_leading_headings(tbl_path.read_text(encoding="utf-8").strip())
                result.append(f"**Table {n}**\n\n{tbl_content}")
            else:
                logger.warning("Table file not found: %s", tbl_path)
            inserted_tbls.add(n)

    return "\n\n".join(result)


def _inline_table_references(text: str) -> str:
    """Replace CSV file references with actual table content from .md files.

    Finds patterns like ``*See `paper/tables/TABLE.csv` for full results.*``
    and replaces them with the corresponding .md table content.

    Parameters
    ----------
    text : str
        Markdown content potentially containing CSV references.

    Returns
    -------
    str
        Markdown with CSV references replaced by inline table content.
    """
    # Map known CSV references to correct .md files where names differ
    _CSV_TO_MD: dict[str, str] = {}

    def _resolve_csv_ref(match: re.Match[str]) -> str:
        csv_name = match.group(1)
        # Strip extension to get base name
        base = csv_name.removesuffix(".csv")
        # Check for remapped names
        md_base = _CSV_TO_MD.get(base, base)
        md_path = PAPER_DIR / "tables" / f"{md_base}.md"
        if md_path.exists():
            # Strip heading lines from table files (they duplicate context headings)
            return _strip_leading_headings(md_path.read_text(encoding="utf-8").strip())
        logger.warning("Table file not found for inline: %s", md_path)
        return match.group(0)

    return re.sub(
        r"\*See `paper/tables/([^`]+\.csv)` for full results\.\*",
        _resolve_csv_ref,
        text,
    )


def _insert_introduction_heading(text: str) -> str:
    """Insert a ``## Introduction`` heading after the abstract (review copy only).

    Nature Articles run the main text straight on after the abstract with no
    heading, which makes the combined review copy read as if the abstract were
    many paragraphs long. For the review document only, this inserts a single
    ``## Introduction`` heading between the abstract paragraph and the first body
    paragraph. No-op if the structure is absent or the heading already present.
    """
    paras = re.split(r"\n\n+", text)
    for i, para in enumerate(paras):
        if para.strip().startswith("## Abstract"):
            insert_at = i + 2  # heading(i), abstract body(i+1), then intro(i+2)
            if insert_at < len(paras) and not paras[insert_at].strip().startswith(
                "## Introduction"
            ):
                paras.insert(insert_at, "## Introduction")
            return "\n\n".join(paras)
    return text


def _preprocess_review_markdown() -> str:
    """Preprocess and combine all paper sections for review.

    Concatenates skeleton.md (with Extended Data section stripped),
    extended_data.md, and supplementary_information.md with page-break
    separators into a single document. The Extended Data section is
    stripped from skeleton.md because extended_data.md provides the
    full content with actual figures and tables.

    CSV file references (``*See `paper/tables/...`*``) are replaced
    with the actual table content from the corresponding .md files.

    Returns
    -------
    str
        Combined preprocessed markdown content.
    """
    parts: list[str] = []

    # Main manuscript — strip Extended Data section (it's in extended_data.md)
    skeleton = DEFAULT_SKELETON
    if skeleton.exists():
        text = skeleton.read_text(encoding="utf-8")
        # Strip Extended Data section from the end of skeleton
        ed_marker = "\n## Extended Data\n"
        if ed_marker in text:
            text = text[: text.index(ed_marker)].rstrip() + "\n"
        text = _insert_introduction_heading(text)
        text = _insert_main_figures_and_tables(text)
        parts.append(_resolve_figure_paths(text))

    # Extended Data — inline table content from .md files
    ed_md = PAPER_DIR / "extended_data.md"
    if ed_md.exists():
        text = ed_md.read_text(encoding="utf-8")
        text = _inline_table_references(text)
        parts.append(_resolve_figure_paths(text))

    # Supplementary Information — inline table content from .md files
    si_md = PAPER_DIR / "supplementary_information.md"
    if si_md.exists():
        text = si_md.read_text(encoding="utf-8")
        text = _inline_table_references(text)
        parts.append(_resolve_figure_paths(text))

    pagebreak = '\n\n```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n\n'
    return pagebreak.join(parts)


def _create_reference_docx(ref_path: Path, *, line_spacing: float = LINE_SPACING) -> None:
    """Generate a pandoc reference.docx with Nature Water styling.

    Parameters
    ----------
    ref_path : Path
        Output path for the reference template.
    line_spacing : float
        Multiple line-spacing applied to the Normal style (1.5 for submission
        docs, 1.0 for the single-spaced review copy).
    """
    from docx import Document
    from docx.enum.text import WD_LINE_SPACING
    from docx.shared import Inches, Pt

    doc = Document()

    # Set default font
    style = doc.styles["Normal"]
    font = style.font
    font.name = FONT_NAME
    font.size = Pt(FONT_SIZE_PT)

    # Set line spacing
    paragraph_format = style.paragraph_format
    paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    paragraph_format.line_spacing = line_spacing

    # Set margins
    for section in doc.sections:
        section.top_margin = Inches(MARGIN_INCHES)
        section.bottom_margin = Inches(MARGIN_INCHES)
        section.left_margin = Inches(MARGIN_INCHES)
        section.right_margin = Inches(MARGIN_INCHES)

    # Heading styles
    for level in range(1, 4):
        style_name = f"Heading {level}"
        if style_name in doc.styles:
            h_style = doc.styles[style_name]
            h_style.font.name = FONT_NAME
            h_style.font.bold = True
            if level == 1:
                h_style.font.size = Pt(14)
            elif level == 2:
                h_style.font.size = Pt(12)
            else:
                h_style.font.size = Pt(11)

    doc.save(str(ref_path))
    logger.info("Created reference template: %s", ref_path)


def _convert_with_pandoc(
    md_content: str,
    output_path: Path,
    ref_docx: Path,
) -> bool:
    """Convert markdown to docx using pandoc.

    Parameters
    ----------
    md_content : str
        Preprocessed markdown content.
    output_path : Path
        Output .docx path.
    ref_docx : Path
        Reference template for styling.

    Returns
    -------
    bool
        True if conversion succeeded.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(md_content)
        tmp_md = Path(f.name)

    try:
        cmd = [
            "pandoc",
            str(tmp_md),
            "-o",
            str(output_path),
            f"--reference-doc={ref_docx}",
            # Be explicit about the extensions the content relies on (all enabled
            # by default in pandoc 3.x, but pinned here against future changes):
            # implicit_figures -> captioned figures with width attrs; raw_attribute
            # -> the {=openxml} page-break blocks; pipe_tables -> GFM tables.
            "--from=markdown+implicit_figures+raw_attribute+pipe_tables+tex_math_dollars",
            "--to=docx",
            "--standalone",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.warning("Pandoc failed: %s", result.stderr)
            return False
        logger.info("Pandoc conversion succeeded: %s", output_path)
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.warning("Pandoc conversion failed: %s", exc)
        return False
    finally:
        tmp_md.unlink(missing_ok=True)


def _postprocess_docx(docx_path: Path, *, line_spacing: float = LINE_SPACING) -> None:
    """Post-process the generated docx with python-docx.

    Ensures consistent font, margins, and line spacing across body paragraphs
    *and* table cells, and shrinks the cell font of very wide tables (more than
    :data:`WIDE_TABLE_COL_THRESHOLD` columns) so the per-analyte and full
    benchmark tables stay on the page (pandoc does not auto-shrink fonts).

    Parameters
    ----------
    docx_path : Path
        Path to the docx file to post-process.
    line_spacing : float
        Multiple line-spacing to apply (1.5 for submission docs, 1.0 for the
        single-spaced review copy).
    """
    from docx import Document
    from docx.enum.text import WD_LINE_SPACING
    from docx.shared import Inches, Pt

    doc = Document(str(docx_path))

    # Ensure margins
    for section in doc.sections:
        section.top_margin = Inches(MARGIN_INCHES)
        section.bottom_margin = Inches(MARGIN_INCHES)
        section.left_margin = Inches(MARGIN_INCHES)
        section.right_margin = Inches(MARGIN_INCHES)

    def _apply(paragraph, *, cell_font_pt: float | None = None) -> None:
        for run in paragraph.runs:
            run.font.name = FONT_NAME
            if cell_font_pt is not None:
                run.font.size = Pt(cell_font_pt)
        paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        paragraph.paragraph_format.line_spacing = line_spacing

    # Body paragraphs (font name + spacing only — never touch heading sizes).
    for paragraph in doc.paragraphs:
        _apply(paragraph)

    # Table cells: same font/spacing, plus a smaller font for very wide tables.
    for table in doc.tables:
        cell_font_pt = (
            WIDE_TABLE_FONT_SIZE_PT if len(table.columns) > WIDE_TABLE_COL_THRESHOLD else None
        )
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    _apply(paragraph, cell_font_pt=cell_font_pt)

    doc.save(str(docx_path))
    logger.info("Post-processed: %s", docx_path)


# Inline markdown tokenizer. Ordered so ``**`` is matched before single-star
# italics and intra-word ``_`` / superscript carets (``^1^``) stay literal.
_INLINE_PATTERN = re.compile(
    r"(?P<code>`[^`]+`)"
    r"|(?P<bold>\*\*[^*]+\*\*)"
    r"|(?P<link>\[[^\]]+\]\([^)]+\))"
    r"|(?P<istar>(?<![\w*])\*(?!\s)[^*\n]+?\*(?![\w*]))"
    r"|(?P<iunder>(?<![\w_])_(?!\s)[^_\n]+?_(?![\w_]))"
)


# Backslash-escaped markdown punctuation (e.g. ``\*`` -> literal ``*``) is mapped to
# private-use sentinels before tokenizing so an escaped ``*`` inside ``**bold**`` does not
# break the span; each sentinel is restored to its literal character on every emitted run.
_ESCAPABLE = "*_`#~()[]\\"
_ESCAPE_SENTINELS = {ch: chr(0xE000 + i) for i, ch in enumerate(_ESCAPABLE)}
_UNESCAPE_TABLE = {ord(sentinel): ch for ch, sentinel in _ESCAPE_SENTINELS.items()}
_ESCAPE_RE = re.compile(r"\\([" + re.escape(_ESCAPABLE) + r"])")


def _emit_run(paragraph, text: str, *, bold: bool, italic: bool, code: bool) -> None:
    """Append one run, restoring escaped characters and applying active styles."""
    if not text:
        return
    run = paragraph.add_run(text.translate(_UNESCAPE_TABLE))
    if bold:
        run.bold = True
    if italic:
        run.italic = True
    if code:
        run.font.name = "Consolas"


def _tokenize_inline(paragraph, text: str, *, bold: bool, italic: bool) -> None:
    """Recursively tokenize inline markdown so code/emphasis nest inside a span.

    A code span or emphasis nested inside ``**bold**`` / ``*italic*`` keeps both
    styles (e.g. ``\\`path\\``` inside ``*...*`` renders monospace-italic instead
    of leaking literal backticks).
    """
    pos = 0
    for m in _INLINE_PATTERN.finditer(text):
        if m.start() > pos:
            _emit_run(paragraph, text[pos : m.start()], bold=bold, italic=italic, code=False)
        kind, token = m.lastgroup, m.group()
        if kind == "code":
            _emit_run(paragraph, token[1:-1], bold=bold, italic=italic, code=True)
        elif kind == "bold":
            _tokenize_inline(paragraph, token[2:-2], bold=True, italic=italic)
        elif kind == "link":
            lm = re.match(r"\[([^\]]+)\]\([^)]+\)", token)
            _emit_run(
                paragraph, lm.group(1) if lm else token, bold=bold, italic=italic, code=False
            )
        else:  # istar / iunder
            _tokenize_inline(paragraph, token[1:-1], bold=bold, italic=True)
        pos = m.end()
    if pos < len(text):
        _emit_run(paragraph, text[pos:], bold=bold, italic=italic, code=False)


def _add_inline_runs(paragraph, text: str) -> None:
    """Add markdown inline-formatted runs to a paragraph.

    Tokenizes ``**bold**``, ``*italic*`` / ``_italic_``, ``\\`code\\``` and
    ``[text](url)`` (link text only) into python-docx runs; unmatched text
    becomes plain runs. Spans nest (code/emphasis inside another span keep both
    styles), backslash escapes (``\\*``) render as the literal character, and
    superscript carets (``^1^``) and stray asterisks are emitted literally and
    never raise.

    Parameters
    ----------
    paragraph : docx.text.paragraph.Paragraph
        Target paragraph to append runs to.
    text : str
        Raw inline markdown for a single logical line / cell (no newlines).
    """
    escaped = _ESCAPE_RE.sub(lambda m: _ESCAPE_SENTINELS[m.group(1)], text)
    _tokenize_inline(paragraph, escaped, bold=False, italic=False)


def _is_table_separator(row: str) -> bool:
    """Return True for a GFM alignment row (only ``|:- `` chars, has a dash)."""
    body = row.strip().strip("|")
    return bool(body) and "-" in body and set(body) <= set("|:- \t")


def _split_table_row(row: str) -> list[str]:
    """Split a markdown pipe row into stripped cell strings."""
    return [c.strip() for c in row.strip().strip("|").split("|")]


def _add_markdown_table(doc, rows: list[str]) -> None:
    """Render a buffered markdown pipe-table block as a real Word table.

    The GFM alignment row (``|:---|---:|``) is dropped, the header row is bold,
    body cells get inline formatting, ragged rows are padded/truncated to the
    header column count, and very wide tables shrink their font so the 44-column
    per-analyte table renders without raising.

    Parameters
    ----------
    doc : docx.document.Document
        Target document.
    rows : list[str]
        Consecutive raw lines whose stripped form starts with ``|``.
    """
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt

    cells = [_split_table_row(r) for r in rows if r.strip() and not _is_table_separator(r)]
    if not cells:
        return
    header = cells[0]
    body = cells[1:]
    ncols = len(header)
    if ncols == 0:
        return

    table = doc.add_table(rows=1, cols=ncols)
    if "Table Grid" in doc.styles:
        table.style = "Table Grid"
    table.autofit = True
    font_pt = WIDE_TABLE_FONT_SIZE_PT if ncols > WIDE_TABLE_COL_THRESHOLD else TABLE_FONT_SIZE_PT

    def _fill(target_cells, values, *, bold: bool) -> None:
        padded = (values + [""] * ncols)[:ncols]
        for cell, val in zip(target_cells, padded, strict=False):
            para = cell.paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT
            if bold:
                run = para.add_run(val)
                run.bold = True
                run.font.size = Pt(font_pt)
            else:
                _add_inline_runs(para, val)
                for run in para.runs:
                    run.font.size = Pt(font_pt)

    _fill(table.rows[0].cells, header, bold=True)
    for body_row in body:
        _fill(table.add_row().cells, body_row, bold=False)


def _setup_document(doc, *, line_spacing: float) -> None:
    """Apply margins, default font, and line spacing to a Document in place.

    Parameters
    ----------
    doc : docx.document.Document
        Document to configure.
    line_spacing : float
        Multiple line-spacing applied to the Normal style.
    """
    from docx.enum.text import WD_LINE_SPACING
    from docx.shared import Inches, Pt

    for section in doc.sections:
        section.top_margin = Inches(MARGIN_INCHES)
        section.bottom_margin = Inches(MARGIN_INCHES)
        section.left_margin = Inches(MARGIN_INCHES)
        section.right_margin = Inches(MARGIN_INCHES)

    style = doc.styles["Normal"]
    style.font.name = FONT_NAME
    style.font.size = Pt(FONT_SIZE_PT)
    style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    style.paragraph_format.line_spacing = line_spacing
    style.paragraph_format.space_after = Pt(6)  # modest, journal-like separation


def _render_markdown_to_doc(doc, md_content: str, *, image_base: Path = PAPER_DIR) -> None:
    """Render markdown into an existing Document (shared fallback core).

    Handles headings, images, fenced code blocks (including the ``{=openxml}``
    page-break sentinel), pipe tables, bullet/numbered lists, ``---`` page
    breaks, and inline formatting. Margins / Normal style are configured by the
    caller via :func:`_setup_document`.

    Parameters
    ----------
    doc : docx.document.Document
        Document to populate.
    md_content : str
        Markdown text.
    image_base : Path, optional
        Base directory for resolving relative image paths. An absolute path in
        the markdown overrides this (so already-resolved review paths work too).
    """
    from docx.shared import Inches, Pt

    image_exts = (".png", ".jpg", ".jpeg")
    table_buf: list[str] = []
    fence_buf: list[str] = []
    prose_buf: list[str] = []
    list_buf: list[str] = []
    list_style: str | None = None
    in_fence = False
    fence_info = ""

    def _flush_table() -> None:
        if table_buf:
            _add_markdown_table(doc, list(table_buf))
            table_buf.clear()

    def _flush_fence() -> None:
        joined = "\n".join(fence_buf)
        if fence_info.strip() == "{=openxml}" and 'w:type="page"' in joined:
            doc.add_page_break()
        elif joined.strip():
            run = doc.add_paragraph().add_run(joined)
            run.font.name = "Consolas"
            run.font.size = Pt(9)
        fence_buf.clear()

    def _flush_prose() -> None:
        # Soft-wrapped source lines reflow into a single paragraph.
        if prose_buf:
            _add_inline_runs(doc.add_paragraph(), " ".join(prose_buf))
            prose_buf.clear()

    def _flush_list() -> None:
        # A list item (with any hard-wrapped continuation lines) is one paragraph.
        nonlocal list_style
        if list_buf:
            _add_inline_runs(doc.add_paragraph(style=list_style), " ".join(list_buf))
            list_buf.clear()
        list_style = None

    def _flush_block() -> None:
        _flush_prose()
        _flush_list()

    for line in md_content.split("\n"):
        stripped = line.strip()

        # Fenced code block state machine (handles {=openxml} page breaks too).
        if stripped.startswith("```"):
            if not in_fence:
                _flush_block()
                _flush_table()
                in_fence = True
                fence_info = stripped[3:]
                fence_buf.clear()
            else:
                in_fence = False
                _flush_fence()
                fence_info = ""
            continue
        if in_fence:
            fence_buf.append(line)
            continue

        # Buffer contiguous pipe-table rows; any other line flushes the buffer.
        if stripped.startswith("|") and "|" in stripped[1:]:
            _flush_block()
            table_buf.append(stripped)
            continue
        _flush_table()

        if stripped == "":
            # Blank line is a hard paragraph boundary (ends prose and any open list item).
            _flush_block()
        elif stripped == "---":
            _flush_block()
            doc.add_page_break()
        elif stripped.startswith("# "):
            _flush_block()
            doc.add_heading(stripped[2:], level=1)
        elif stripped.startswith("## "):
            _flush_block()
            doc.add_heading(stripped[3:], level=2)
        elif stripped.startswith("### "):
            _flush_block()
            doc.add_heading(stripped[4:], level=3)
        elif stripped.startswith("!["):
            _flush_block()
            match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
            if match:
                alt_text = match.group(1)
                img_path = image_base / match.group(2)
                if img_path.exists() and img_path.suffix.lower() in image_exts:
                    try:
                        doc.add_picture(str(img_path), width=Inches(5.5))
                    except Exception:
                        doc.add_paragraph().add_run(f"[{alt_text}]").italic = True
                else:
                    doc.add_paragraph().add_run(f"[{alt_text}]").italic = True
        elif stripped.startswith(("- ", "* ")):
            # New list item: close prose and the previous item, then open this one.
            _flush_prose()
            _flush_list()
            list_style = "List Bullet"
            list_buf.append(stripped[2:])
        elif num_match := re.match(r"^\d+\.\s+", stripped):
            _flush_prose()
            _flush_list()
            list_style = "List Number"
            list_buf.append(stripped[num_match.end() :])
        elif list_style is not None:
            # Hard-wrapped continuation of the open list item.
            list_buf.append(stripped)
        else:
            prose_buf.append(stripped)

    _flush_block()
    _flush_table()
    if in_fence:
        _flush_fence()


def _convert_pure_docx(md_path: Path, output_path: Path) -> None:
    """Build docx programmatically from markdown (fallback).

    Parameters
    ----------
    md_path : Path
        Path to the markdown source file.
    output_path : Path
        Output .docx path.
    """
    from docx import Document

    text = md_path.read_text(encoding="utf-8")
    doc = Document()
    _setup_document(doc, line_spacing=LINE_SPACING)

    # Strip Extended Data from main manuscript only (not from extended_data.md itself)
    ed_marker = "### Extended Data Fig."
    if md_path.stem == "skeleton" and ed_marker in text:
        text = text[: text.index(ed_marker)].rstrip()

    _render_markdown_to_doc(doc, text, image_base=PAPER_DIR)
    doc.save(str(output_path))
    logger.info("Pure python-docx conversion: %s", output_path)


def _convert_pure_docx_from_string(md_content: str, output_path: Path) -> None:
    """Build docx programmatically from a markdown string (fallback for review).

    Uses true single spacing so the combined review copy reads like a
    final-formatted journal article rather than a double-spaced draft.

    Parameters
    ----------
    md_content : str
        Preprocessed markdown content (with absolute image paths).
    output_path : Path
        Output .docx path.
    """
    from docx import Document

    doc = Document()
    _setup_document(doc, line_spacing=REVIEW_LINE_SPACING)
    _render_markdown_to_doc(doc, md_content, image_base=PAPER_DIR)
    doc.save(str(output_path))
    logger.info("Pure python-docx conversion (review): %s", output_path)


def _embed_provenance(docx_path: Path) -> None:
    """Embed the G12 source-manifest hash + display provenance into core properties.

    The gate (final_gate.py G12) recomputes the manifest hash from the working tree and
    compares it to the embedded value — content-addressed, so it stays verifiable after
    unrelated commits (an embedded-git-SHA==HEAD predicate could never pass with
    git-tracked DOCX). The git SHA and timestamp are display-only provenance, visible in
    Word under File -> Info (kept out of the document body so submission artifacts carry
    no extra text). No-op for outputs without a DERIVATIONS.tsv G12 manifest (e.g. tests
    writing to temp dirs).
    """
    import datetime
    import subprocess

    try:
        from paper import gate_lib
    except ImportError:  # pragma: no cover - script-dir-on-path fallback
        import gate_lib

    from docx import Document

    try:
        artifact = docx_path.resolve().relative_to(gate_lib.REPO).as_posix()
    except ValueError:
        return  # outside the repo: no manifest defined
    rows = [r for r in gate_lib.parse_derivations() if r.check == "G12" and r.artifact == artifact]
    if not rows:
        return
    src_hash = gate_lib.manifest_hash(artifact)
    try:
        git_sha = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=gate_lib.REPO,
            ).stdout.strip()
            or "unknown"
        )
    except OSError:  # pragma: no cover - git absent
        git_sha = "unknown"
    generated = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    doc = Document(str(docx_path))
    doc.core_properties.comments = gate_lib.provenance_string(src_hash, git_sha, generated)
    doc.save(str(docx_path))
    logger.info("Embedded provenance (source_sha256=%s...) into %s", src_hash[:12], docx_path)


def convert_review_to_docx(output_path: Path) -> Path:
    """Combine all paper sections into a single review docx.

    Parameters
    ----------
    output_path : Path
        Output .docx path.

    Returns
    -------
    Path
        Path to the generated .docx file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    md_content = _preprocess_review_markdown()

    if _has_pandoc():
        logger.info("Using pandoc for review conversion")
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as ref_tmp:
            ref_path = Path(ref_tmp.name)

        try:
            _create_reference_docx(ref_path, line_spacing=REVIEW_LINE_SPACING)
            if _convert_with_pandoc(md_content, output_path, ref_path):
                _postprocess_docx(output_path, line_spacing=REVIEW_LINE_SPACING)
                _embed_provenance(output_path)
                return output_path
        finally:
            ref_path.unlink(missing_ok=True)

        logger.info("Pandoc failed, falling back to pure python-docx")

    _convert_pure_docx_from_string(md_content, output_path)
    _embed_provenance(output_path)
    return output_path


def convert_primer_to_docx(primer_path: Path, output_path: Path) -> Path:
    """Convert a standalone plain-language primer markdown to a single-spaced docx.

    Mirrors :func:`convert_review_to_docx` (single spacing for a compact, readable
    companion document) but reads the markdown file directly, skipping the
    figure/table injection and Extended-Data handling that
    :func:`convert_markdown_to_docx` applies to the manuscript. Intended for a
    short, self-contained reader's guide that references no main figures or tables.

    Parameters
    ----------
    primer_path : Path
        Input primer markdown file.
    output_path : Path
        Output .docx path.

    Returns
    -------
    Path
        Path to the generated .docx file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    md_content = primer_path.read_text(encoding="utf-8")

    if _has_pandoc():
        logger.info("Using pandoc for primer conversion")
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as ref_tmp:
            ref_path = Path(ref_tmp.name)

        try:
            _create_reference_docx(ref_path, line_spacing=REVIEW_LINE_SPACING)
            if _convert_with_pandoc(md_content, output_path, ref_path):
                _postprocess_docx(output_path, line_spacing=REVIEW_LINE_SPACING)
                _embed_provenance(output_path)
                return output_path
        finally:
            ref_path.unlink(missing_ok=True)

        logger.info("Pandoc failed, falling back to pure python-docx")

    _convert_pure_docx_from_string(md_content, output_path)
    _embed_provenance(output_path)
    return output_path


def convert_markdown_to_docx(
    md_path: Path,
    output_path: Path,
) -> Path:
    """Convert a markdown file to Word format.

    Tries pandoc first, falls back to pure python-docx.

    Parameters
    ----------
    md_path : Path
        Input markdown file.
    output_path : Path
        Output .docx path.

    Returns
    -------
    Path
        Path to the generated .docx file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if _has_pandoc():
        logger.info("Using pandoc for conversion")
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as ref_tmp:
            ref_path = Path(ref_tmp.name)

        try:
            _create_reference_docx(ref_path)
            md_content = _preprocess_markdown(md_path)
            if _convert_with_pandoc(md_content, output_path, ref_path):
                _postprocess_docx(output_path)
                _embed_provenance(output_path)
                return output_path
        finally:
            ref_path.unlink(missing_ok=True)

        logger.info("Pandoc failed, falling back to pure python-docx")

    _convert_pure_docx(md_path, output_path)
    _embed_provenance(output_path)
    return output_path


def main() -> None:
    """CLI entry point for Word conversion."""
    import click

    @click.command()
    @click.option(
        "--skeleton",
        default=str(DEFAULT_SKELETON),
        help="Path to main manuscript markdown.",
    )
    @click.option(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output .docx path.",
    )
    @click.option(
        "--extended-data",
        is_flag=True,
        help="Also convert Extended Data document.",
    )
    @click.option(
        "--supplementary",
        is_flag=True,
        help="Also convert Supplementary Information document.",
    )
    @click.option(
        "--all-docs",
        is_flag=True,
        help="Convert all documents (manuscript + ED + SI).",
    )
    @click.option(
        "--review",
        is_flag=True,
        help="Generate a single combined review document (manuscript + ED + SI).",
    )
    @click.option(
        "--primer",
        is_flag=True,
        help="Generate only the one-page plain-language primer (does not touch the manuscript).",
    )
    @click.option("-v", "--verbose", is_flag=True, help="Verbose logging.")
    def cli(
        skeleton: str,
        output: str,
        extended_data: bool,
        supplementary: bool,
        all_docs: bool,
        review: bool,
        primer: bool,
        verbose: bool,
    ) -> None:
        """Convert AquaContam paper markdown to Word format."""
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

        if review:
            review_out = PAPER_DIR / "manuscript_review.docx"
            click.echo(f"Generating combined review document -> {review_out.name}")
            convert_review_to_docx(review_out)
            click.echo(f"  Written: {review_out}")
            click.echo("Done!")
            return

        if primer:
            primer_md = PAPER_DIR / "plain_language_primer.md"
            primer_out = PAPER_DIR / "plain_language_primer.docx"
            if not primer_md.exists():
                click.echo(f"  Skipping primer: {primer_md} not found")
                return
            click.echo(f"Generating plain-language primer -> {primer_out.name}")
            convert_primer_to_docx(primer_md, primer_out)
            click.echo(f"  Written: {primer_out}")
            click.echo("Done!")
            return

        if all_docs:
            extended_data = True
            supplementary = True

        # Main manuscript
        md_path = Path(skeleton)
        out_path = Path(output)
        click.echo(f"Converting {md_path.name} -> {out_path.name}")
        convert_markdown_to_docx(md_path, out_path)
        click.echo(f"  Written: {out_path}")

        # Extended Data
        if extended_data:
            ed_md = PAPER_DIR / "extended_data.md"
            ed_out = out_path.parent / "extended_data.docx"
            if ed_md.exists():
                click.echo(f"Converting {ed_md.name} -> {ed_out.name}")
                convert_markdown_to_docx(ed_md, ed_out)
                click.echo(f"  Written: {ed_out}")
            else:
                click.echo(f"  Skipping Extended Data: {ed_md} not found")

        # Supplementary Information
        if supplementary:
            si_md = PAPER_DIR / "supplementary_information.md"
            si_out = out_path.parent / "supplementary_information.docx"
            if si_md.exists():
                click.echo(f"Converting {si_md.name} -> {si_out.name}")
                convert_markdown_to_docx(si_md, si_out)
                click.echo(f"  Written: {si_out}")
            else:
                click.echo(f"  Skipping SI: {si_md} not found")

        click.echo("Done!")

    cli()


if __name__ == "__main__":
    main()
