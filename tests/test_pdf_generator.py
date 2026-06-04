"""Tests for pdf_generator edge cases.

These run without an LLM — they just feed crafted markdown and assert
that the renderer never crashes and produces a valid PDF.
"""
import os
import tempfile
import pytest
from fpdf import FPDF

from pdf_generator import (
    generate_pdf_from_markdown,
    _safe_text,
    _normalize_table_row,
    _is_table_separator,
    _render_table,
)


# ──────────────────────────────────────────────────────────────
#  _safe_text — unicode sanitization
# ──────────────────────────────────────────────────────────────
class TestSafeText:
    def test_empty(self):
        assert _safe_text("") == ""
        assert _safe_text(None) == ""

    def test_ascii_passthrough(self):
        assert _safe_text("Hello World 123") == "Hello World 123"

    def test_latin1_passthrough(self):
        assert _safe_text("café résumé") == "café résumé"

    def test_unicode_replaced(self):
        # Em-dash should become a hyphen
        assert _safe_text("a — b") == "a - b"
        # Smart quotes
        assert _safe_text("\u201chello\u201d") == '"hello"'
        # Ellipsis
        assert _safe_text("wait\u2026") == "wait..."

    def test_unknown_codepoint_replaced_with_question(self):
        # CJK — not in our transliteration table
        assert _safe_text("日本") == "??"


# ──────────────────────────────────────────────────────────────
#  Table helpers
# ──────────────────────────────────────────────────────────────
class TestTableHelpers:
    def test_normalize_simple(self):
        assert _normalize_table_row("| a | b | c |") == ["a", "b", "c"]

    def test_normalize_no_outer_pipes(self):
        assert _normalize_table_row("a | b | c") == ["a", "b", "c"]

    def test_normalize_with_padding(self):
        assert _normalize_table_row("  |  a  |  b  |  ") == ["a", "b"]

    def test_separator_detection(self):
        assert _is_table_separator(["---", ":---:", "---:"])
        assert _is_table_separator(["-", "-", "-"])
        assert not _is_table_separator(["a", "b", "c"])
        assert not _is_table_separator([])

    def test_render_empty_noop(self):
        pdf = FPDF()
        pdf.add_page()
        _render_table(pdf, [])
        _render_table(pdf, [[]])
        # Should not raise and shouldn't add a page
        assert pdf.page_no() == 1


# ──────────────────────────────────────────────────────────────
#  Full PDF generation
# ──────────────────────────────────────────────────────────────
def _gen(md, company="Test Co"):
    """Helper: render markdown to a temp PDF and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    generate_pdf_from_markdown(md, company, tmp.name)
    return tmp.name


def _pdf_valid(path):
    return os.path.exists(path) and os.path.getsize(path) > 100


class TestFullRender:
    def test_basic_report(self):
        md = """# Market Intelligence Report: Vercel

## Executive Summary
Vercel is **leading** the edge platform space.

## Top Competitors

| Metric | Vercel | Netlify |
| --- | --- | --- |
| Founded | 2015 | 2014 |
| Pricing | Freemium | Freemium |

### Strengths
- **DX**: Best-in-class developer experience
- **Performance**: Edge-native
"""
        path = _gen(md)
        try:
            assert _pdf_valid(path)
        finally:
            os.remove(path)

    def test_empty_markdown_raises(self):
        with pytest.raises(ValueError):
            _gen("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            _gen("   \n\n   \t  \n")

    def test_ragged_table_does_not_crash(self):
        # LLMs occasionally produce tables with mismatched column counts.
        md = """
| H1 | H2 | H3 |
| --- | --- | --- |
| a | b |
| a | b | c | d | e |
"""
        path = _gen(md)
        try:
            assert _pdf_valid(path)
        finally:
            os.remove(path)

    def test_fenced_code_block(self):
        md = """
## Example

```python
def hello():
    return "world"
```

End.
"""
        path = _gen(md)
        try:
            assert _pdf_valid(path)
        finally:
            os.remove(path)

    def test_horizontal_rule(self):
        md = "## A\n\n---\n\n## B"
        path = _gen(md)
        try:
            assert _pdf_valid(path)
        finally:
            os.remove(path)

    def test_nested_lists(self):
        md = """
- Level 1
  - Level 2
    - Level 3
- Back to 1
"""
        path = _gen(md)
        try:
            assert _pdf_valid(path)
        finally:
            os.remove(path)

    def test_numbered_list(self):
        md = """
1. First item
2. Second item
3. Third item
"""
        path = _gen(md)
        try:
            assert _pdf_valid(path)
        finally:
            os.remove(path)

    def test_inline_code(self):
        md = "Run `npm install` to get started."
        path = _gen(md)
        try:
            assert _pdf_valid(path)
        finally:
            os.remove(path)

    def test_inline_bold_and_italic_markers(self):
        md = "Some **bold** text and *italic* text."
        path = _gen(md)
        try:
            assert _pdf_valid(path)
        finally:
            os.remove(path)

    def test_unbalanced_bold_does_not_crash(self):
        # Unbalanced ** — common LLM artifact
        md = "This has **unbalanced bold that never closes."
        path = _gen(md)
        try:
            assert _pdf_valid(path)
        finally:
            os.remove(path)

    def test_unicode_content(self):
        md = "# Café résumé 日本 — edge cases ✓"
        path = _gen(md)
        try:
            assert _pdf_valid(path)
        finally:
            os.remove(path)

    def test_trailing_table(self):
        md = """## Final

| Col1 | Col2 |
| --- | --- |
| a | b |
| c | d |"""
        path = _gen(md)
        try:
            assert _pdf_valid(path)
        finally:
            os.remove(path)

    def test_output_dir_created(self):
        with tempfile.TemporaryDirectory() as td:
            nested = os.path.join(td, "nested", "subdir", "report.pdf")
            generate_pdf_from_markdown("# Hello", "Test", nested)
            assert os.path.exists(nested)

    def test_multi_page_report(self):
        # Force many pages to test header/footer behavior
        md = "# Title\n\n" + ("## Section\n\nLorem ipsum dolor sit amet. " * 50 + "\n\n") * 20
        path = _gen(md)
        try:
            assert _pdf_valid(path)
            # fpdf2 produces %PDF-... headers
            with open(path, "rb") as f:
                assert f.read(4) == b"%PDF"
        finally:
            os.remove(path)

    def test_unicode_topic_in_cover_and_header(self):
        # Regression: em-dash (or any non-Latin-1) in the topic name used to
        # crash the cover page and header because those strings bypassed
        # _safe_text. The fix sanitizes title_text/subtitle_text in __init__.
        for name in ["Vercel — Edge Platform", "AI / ML", "café 日本 ✓", "Trailing —"]:
            md = "# Report for " + name + "\n\n## Section\n\nBody text."
            path = _gen(md, company=name)
            try:
                assert _pdf_valid(path)
            finally:
                if os.path.exists(path):
                    os.remove(path)
