"""
Markdown → PDF renderer (fpdf2).

Handles a meaningful subset of CommonMark that the LLM typically produces:
- H1 / H2 / H3 headings with colored accent rules
- Bullet lists ( - or * ), nested up to 3 levels
- GFM-style tables with header rows, separator rows, and ragged column counts
- Paragraphs with inline **bold** spans
- Fenced code blocks (```)
- Horizontal rules (---)
- Inline `code`
- Lines that are too long to fit (auto wrap)
- Unicode that's safe for Helvetica's Latin-1 range (others are transliterated)

Anything unrecognized falls back to a paragraph render — we never crash.
"""

import re
import os
import unicodedata
from datetime import datetime
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Helvetica (the only built-in font guaranteed to exist) covers Latin-1.
# Anything outside it gets ASCII-folded so the PDF doesn't blow up.
def _safe_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    out = []
    for ch in s:
        cp = ord(ch)
        if cp < 256:
            out.append(ch)
        else:
            # Best-effort transliteration for common punctuation / symbols
            repl = {
                0x2014: "-", 0x2013: "-", 0x2018: "'", 0x2019: "'",
                0x201C: '"', 0x201D: '"', 0x2026: "...", 0x00B7: "*",
                0x2713: "v", 0x2717: "x", 0x2192: "->", 0x2190: "<-",
                0x2022: "*", 0x25CF: "*", 0x00A0: " ",
            }.get(cp)
            out.append(repl if repl is not None else "?")
    return "".join(out)


class MarketReportPDF(FPDF):
    def __init__(self, title_text, subtitle_text):
        super().__init__()
        # Sanitize once, up front — every render path below then sees only
        # characters that Helvetica's Latin-1 subset can handle.
        self.title_text = _safe_text(title_text or "")
        self.subtitle_text = _safe_text(subtitle_text or "")
        self.set_margins(15, 20, 15)
        self.alias_nb_pages()

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("helvetica", "I", 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 10, f"Market & Competitor Intelligence: {self.title_text}",
                  border=0, new_x=XPos.RIGHT, new_y=YPos.TOP, align="L")
        self.cell(0, 10, datetime.now().strftime("%B %Y"),
                  border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
        self.set_draw_color(226, 232, 240)
        self.line(15, 27, 195, 27)
        self.ln(5)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()} of {{nb}}",
                  border=0, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C")
        self.cell(0, 10, "Confidential - AI Market Analyst Team",
                  border=0, new_x=XPos.RIGHT, new_y=YPos.TOP, align="R")

    def draw_cover_page(self):
        self.add_page()
        self.set_fill_color(15, 23, 42)
        self.rect(0, 0, 210, 120, "F")

        self.set_y(40)
        self.set_font("helvetica", "B", 26)
        self.set_text_color(255, 255, 255)
        self.multi_cell(0, 12, "MARKET & COMPETITOR\nINTELLIGENCE REPORT", align="C")

        self.set_fill_color(99, 102, 241)
        self.rect(80, 75, 50, 3, "F")

        self.set_y(85)
        self.set_font("helvetica", "B", 14)
        self.set_text_color(226, 232, 240)
        self.cell(0, 10, f"Subject: {self.subtitle_text.upper()}",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

        self.set_y(150)
        self.set_font("helvetica", "B", 12)
        self.set_text_color(79, 70, 229)
        self.cell(0, 10, "PREPARED BY", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_font("helvetica", "", 11)
        self.set_text_color(51, 65, 85)
        self.cell(0, 6, "AI Competitor Intelligence & Market Analyst Team",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

        self.ln(10)
        self.set_font("helvetica", "B", 12)
        self.set_text_color(79, 70, 229)
        self.cell(0, 10, "DATE OF ISSUE", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_font("helvetica", "", 11)
        self.set_text_color(51, 65, 85)
        self.cell(0, 6, datetime.now().strftime("%B %d, %Y"),
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

        self.set_y(-30)
        self.set_font("helvetica", "I", 9)
        self.set_text_color(100, 116, 139)
        self.multi_cell(
            0, 5,
            "This report contains automated research and analysis gathered by "
            "autonomous web-crawling agents. The contents are generated using "
            "generative AI models and Tavily search APIs.",
            align="C",
        )


# ──────────────────────────────────────────────────────────────────────
#  Rich-text helpers
# ──────────────────────────────────────────────────────────────────────
_BOLD_SPLIT = re.compile(r"(\*\*[^*]+\*\*)")
_CODE_SPLIT = re.compile(r"(`[^`]+`)")


def _write_rich(pdf, text, h, indent_after=False):
    """Write text with **bold** and `code` inline styling."""
    # Tokenize on ** then ` (bold first to avoid eating backticks inside bold)
    parts = _BOLD_SPLIT.split(text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) >= 4:
            pdf.set_font(pdf.font_family, "B")
            pdf.write(h, _safe_text(part[2:-2]))
        else:
            # Handle inline `code`
            for sub in _CODE_SPLIT.split(part):
                if not sub:
                    continue
                if sub.startswith("`") and sub.endswith("`") and len(sub) >= 2:
                    pdf.set_font("Courier", "B")
                    pdf.write(h, _safe_text(sub[1:-1]))
                    pdf.set_font(pdf.font_family, "")
                else:
                    pdf.set_font(pdf.font_family, "")
                    pdf.write(h, _safe_text(sub))
    if indent_after:
        pdf.ln(h + 1)


def _normalize_table_row(line: str):
    """Strip leading/trailing pipes, split, and strip each cell."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def _is_table_separator(cells):
    """`| --- | :---: |` style alignment row."""
    if not cells:
        return False
    for c in cells:
        c = c.strip().strip(":")
        if not re.fullmatch(r"-{1,}", c):
            return False
    return True


def _render_table(pdf, rows):
    """Render a collected list of cell-rows as a styled table.

    Pads/truncates ragged rows to the header column count so a single
    malformed row from the LLM can't crash the renderer.
    """
    if not rows:
        return
    # Drop leading separator rows
    rows = [r for r in rows if not _is_table_separator(r)]
    if not rows:
        return
    col_count = max(len(r) for r in rows)
    if col_count == 0:
        return
    rows = [r + [""] * (col_count - len(r)) for r in rows]

    available_width = 180
    col_width = available_width / col_count
    pdf.ln(4)

    with pdf.table(width=available_width,
                   col_widths=[col_width] * col_count) as t:
        # Header
        hdr = t.row()
        pdf.set_font("helvetica", "B", 10)
        pdf.set_text_color(255, 255, 255)
        pdf.set_fill_color(30, 41, 59)
        for cell in rows[0]:
            hdr.cell(_safe_text(cell))

        # Body
        pdf.set_font("helvetica", "", 9.5)
        pdf.set_text_color(51, 65, 85)
        for i, row in enumerate(rows[1:]):
            r = t.row()
            pdf.set_fill_color(248, 250, 252) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
            for cell in row:
                r.cell(_safe_text(cell))
    pdf.ln(6)


# ──────────────────────────────────────────────────────────────────────
#  Public entry point
# ──────────────────────────────────────────────────────────────────────
def generate_pdf_from_markdown(markdown_text: str, company_name: str, output_path: str):
    """
    Parses a markdown string and generates a polished PDF report.
    Safe against malformed input — unknown syntax falls back to paragraphs.
    """
    if not markdown_text or not markdown_text.strip():
        raise ValueError("Cannot generate PDF: markdown content is empty.")

    pdf = MarketReportPDF(company_name, company_name)
    pdf.draw_cover_page()
    pdf.add_page()

    lines = markdown_text.splitlines()

    in_code = False
    code_buf: list[str] = []
    in_table = False
    table_rows: list[list[str]] = []
    in_list = False

    def flush_table():
        nonlocal in_table, table_rows
        if in_table and table_rows:
            _render_table(pdf, table_rows)
        in_table = False
        table_rows = []

    def flush_code():
        nonlocal in_code, code_buf
        if not in_code:
            return
        pdf.ln(2)
        pdf.set_text_color(30, 41, 59)
        pdf.set_fill_color(241, 245, 249)
        pdf.set_font("helvetica", "", 9)
        # Force a left-margin start so multi_cell has full width to work with
        pdf.set_x(pdf.l_margin)
        for cl in (code_buf or [""]):
            text = _safe_text(cl) or " "
            pdf.multi_cell(0, 5, text, fill=True)
            # multi_cell leaves x at the right margin; reset for the next line
            pdf.set_x(pdf.l_margin)
        # Reset fill so subsequent draws aren't shaded
        pdf.set_fill_color(255, 255, 255)
        pdf.set_font("helvetica", "", 10.5)
        pdf.set_text_color(51, 65, 85)
        pdf.set_x(pdf.l_margin)
        in_code = False
        code_buf = []

    for raw in lines:
        line = raw.rstrip()

        # ── fenced code block ──────────────────────────────
        if line.strip().startswith("```"):
            if in_code:
                flush_code()
            else:
                flush_table()
                in_code = True
                code_buf = []
            continue
        if in_code:
            code_buf.append(line)
            continue

        # ── table ──────────────────────────────────────────
        if line.lstrip().startswith("|"):
            in_table = True
            cells = _normalize_table_row(line)
            if not _is_table_separator(cells):
                table_rows.append(cells)
            continue
        elif in_table:
            flush_table()

        # ── blank line ─────────────────────────────────────
        if not line.strip():
            in_list = False
            pdf.ln(3)
            continue

        # ── horizontal rule ────────────────────────────────
        if re.fullmatch(r"\s*(-{3,}|\*{3,}|_{3,})\s*", line):
            in_list = False
            pdf.set_draw_color(226, 232, 240)
            y = pdf.get_y() + 1
            pdf.line(15, y, 195, y)
            pdf.ln(4)
            continue

        # ── headings ───────────────────────────────────────
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            in_list = False
            level = len(m.group(1))
            text = m.group(2).strip()
            if level == 1:
                pdf.ln(6)
                pdf.set_font("helvetica", "B", 18)
                pdf.set_text_color(15, 23, 42)
                pdf.cell(0, 10, _safe_text(text),
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_draw_color(79, 70, 229)
                pdf.line(15, pdf.get_y() - 1, 195, pdf.get_y() - 1)
                pdf.ln(4)
            elif level == 2:
                pdf.ln(5)
                pdf.set_font("helvetica", "B", 14)
                pdf.set_text_color(30, 41, 59)
                pdf.cell(0, 8, _safe_text(text),
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_draw_color(226, 232, 240)
                pdf.line(15, pdf.get_y() - 1, 100, pdf.get_y() - 1)
                pdf.ln(3)
            else:
                pdf.ln(3)
                pdf.set_font("helvetica", "B", 12 if level == 3 else 11)
                pdf.set_text_color(79, 70, 229)
                pdf.cell(0, 6, _safe_text(text),
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(2)
            continue

        # ── bullet list (supports nested up to 3 levels) ──
        m = re.match(r"^(\s*)([-*])\s+(.*)$", line)
        if m:
            indent = len(m.group(1))
            level = min(3, indent // 2 + 1)
            content = m.group(3)
            bullet = "  -  " if level == 1 else ("    - " if level == 2 else "      - ")
            pdf.set_font("helvetica", "", 10.5)
            pdf.set_text_color(51, 65, 85)
            x_start = pdf.get_x()
            pdf.write(6, bullet)
            pdf.set_x(x_start + 6 + (level - 1) * 4)
            _write_rich(pdf, content, 6, indent_after=True)
            in_list = True
            continue

        # ── numbered list ──────────────────────────────────
        m = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if m:
            pdf.set_font("helvetica", "", 10.5)
            pdf.set_text_color(51, 65, 85)
            pdf.write(6, "  -  ")
            _write_rich(pdf, m.group(1), 6, indent_after=True)
            in_list = True
            continue

        # ── paragraph (fallback) ───────────────────────────
        pdf.set_font("helvetica", "", 10.5)
        pdf.set_text_color(51, 65, 85)
        _write_rich(pdf, line, 6, indent_after=True)
        in_list = False

    # Flush trailing block constructs
    flush_table()
    flush_code()

    # Final fallback in case of empty trailing content: ensure one page exists
    if pdf.page_no() == 0:
        pdf.add_page()

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    pdf.output(output_path)
