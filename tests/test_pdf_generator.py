"""Tests for pdf_generator edge cases.

These run without an LLM — they just feed crafted markdown and assert
that the renderer never crashes and produces a valid PDF.
"""

import os
import tempfile

import pytest
from fpdf import FPDF

from pdf_generator import (
    _is_table_separator,
    _normalize_table_row,
    _render_table,
    _safe_text,
    generate_pdf_from_markdown,
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

    def test_trademark_and_copyright_fold(self):
        # NFKC folds U+2122 (™) to "TM" before our map runs; © and ® are
        # Latin-1, passed through as-is (fpdf2 silently substitutes a "?"
        # glyph for them in the PDF, but they don't crash).
        assert _safe_text("ChatGPT\u2122") == "ChatGPTTM"
        assert _safe_text("\u00a9 2026 Acme") == "\u00a9 2026 Acme"
        assert _safe_text("Brand\u00ae") == "Brand\u00ae"

    def test_currency_fold(self):
        # Euro is U+20AC (mapped to "EUR"); pound/yen are Latin-1
        # and pass through unchanged.
        assert _safe_text("Price: \u20ac99") == "Price: EUR99"
        assert _safe_text("\u00a3 \u00a5") == "\u00a3 \u00a5"

    def test_math_and_arrows_fold(self):
        # Above-256 chars get mapped; ± (U+00B1) is Latin-1, passes through.
        assert _safe_text("a \u2192 b") == "a -> b"
        assert _safe_text("x \u2260 y") == "x != y"
        assert _safe_text("5 \u00b1 1") == "5 \u00b1 1"
        assert _safe_text("\u221e") == "inf"

    def test_unicode_dash_variants_all_become_hyphen(self):
        # Several dash characters should all collapse to ASCII "-"
        for ch in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212":
            assert _safe_text(f"a{ch}b") == "a-b", f"failed for U+{ord(ch):04X}"

    def test_invisible_chars_removed(self):
        # Zero-width / BOM / soft hyphen should be stripped
        for ch in "\u200b\u200c\u200d\ufeff\u00ad":
            assert _safe_text(f"a{ch}b") == "ab", f"failed for U+{ord(ch):04X}"
        # Various unicode spaces collapse to ASCII space
        for ch in "\u00a0\u2002\u2003\u2009\u200a":
            assert _safe_text(f"a{ch}b") == "a b", f"failed for U+{ord(ch):04X}"

    def test_nfkc_normalization(self):
        # Full-width "Hello" should fold to "Hello"
        assert _safe_text("\uff28\uff45\uff4c\uff4c\uff4f") == "Hello"
        # Ligature fi
        assert _safe_text("\ufb01ne") == "fine"


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
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
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

    def test_unicode_in_body_does_not_crash(self):
        # Trademarks, euro, arrows, en-dash, etc. sprinkled through the body
        # should all be folded and the PDF should still be valid.
        md = """
# Report

## Section A

- ChatGPT\u2122 leads the market
- Priced at \u20ac99/month
- Path: a \u2192 b \u2192 c
- Range: 5\u201310 users
- Em-dash for emphasis \u2014 like this
- Star rating: \u2605\u2605\u2605\u2605\u2606

| Symbol | Name |
| --- | --- |
| \u2122 | Trademark |
| \u20ac | Euro |

End.
"""
        path = _gen(md, company="Test \u2014 Co")
        try:
            assert _pdf_valid(path)
        finally:
            if os.path.exists(path):
                os.remove(path)
