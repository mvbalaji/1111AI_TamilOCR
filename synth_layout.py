"""
synth_layout.py — synthetic document layout renderer for P2 dataset tiers.

Tier definitions (from benchmark_spec.py):
  T1  printed_multicolumn  : 1–3 column newspaper/book layout
  T2  tables_forms         : tabular data with ruling lines and headers
  T3  tanglish             : Tamil–English code-mixed text (same line)

All renderers return PIL Images.  Fonts are loaded from the fonts/ directory.
Text is wrapped automatically to fit column widths.

Usage:
  from synth_layout import render_multicolumn, render_table, render_tanglish
  img = render_multicolumn(lines, font_path, columns=2)
  img = render_table(rows, headers, font_path)
  img = render_tanglish(lines, tamil_font, latin_font)
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Optional

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
    raise ImportError("Pillow required: pip install Pillow") from e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PAGE_W = 800
PAGE_H = 1100
MARGIN = 40
BG = (255, 255, 255)
FG = (0, 0, 0)
RULE = (180, 180, 180)
FONT_SIZE = 22
HEADER_SIZE = 26


def _load_font(font_path: Path | str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(font_path), size)
    except OSError:
        return ImageFont.load_default()


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Word-wrap text to fit max_width pixels."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        bbox = font.getbbox(candidate)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


def _line_height(font: ImageFont.FreeTypeFont) -> int:
    bbox = font.getbbox("Ag|")
    return (bbox[3] - bbox[1]) + 4


# ---------------------------------------------------------------------------
# T1: Multi-column layout
# ---------------------------------------------------------------------------

def render_multicolumn(
    lines: list[str],
    font_path: Path | str,
    columns: int = 2,
    font_size: int = FONT_SIZE,
    page_w: int = PAGE_W,
    page_h: int = PAGE_H,
) -> Image.Image:
    """Render text in a 1–3 column newspaper-style layout."""
    columns = max(1, min(3, columns))
    font = _load_font(font_path, font_size)
    lh = _line_height(font)
    gutter = 20
    col_w = (page_w - 2 * MARGIN - (columns - 1) * gutter) // columns

    img = Image.new("RGB", (page_w, page_h), BG)
    draw = ImageDraw.Draw(img)

    # Distribute lines across columns evenly
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(_wrap_text(line, font, col_w))

    rows_per_col = max(1, (page_h - 2 * MARGIN) // lh)
    col_idx = 0
    row_idx = 0
    for text_line in wrapped:
        if row_idx >= rows_per_col:
            col_idx += 1
            row_idx = 0
        if col_idx >= columns:
            break
        x = MARGIN + col_idx * (col_w + gutter)
        y = MARGIN + row_idx * lh
        draw.text((x, y), text_line, font=font, fill=FG)
        row_idx += 1

    # Vertical rules between columns
    for c in range(1, columns):
        rx = MARGIN + c * (col_w + gutter) - gutter // 2
        draw.line([(rx, MARGIN), (rx, page_h - MARGIN)], fill=RULE, width=1)

    return img


# ---------------------------------------------------------------------------
# T2: Table / form layout
# ---------------------------------------------------------------------------

def render_table(
    rows: list[list[str]],
    headers: list[str],
    font_path: Path | str,
    font_size: int = FONT_SIZE,
    page_w: int = PAGE_W,
) -> Image.Image:
    """Render a bordered table with a header row."""
    font = _load_font(font_path, font_size)
    hfont = _load_font(font_path, HEADER_SIZE)
    lh = _line_height(font)
    hlh = _line_height(hfont)

    n_cols = max(len(headers), max((len(r) for r in rows), default=1))
    col_w = (page_w - 2 * MARGIN) // n_cols
    row_h = lh + 8
    header_h = hlh + 8

    total_h = MARGIN + header_h + len(rows) * row_h + MARGIN
    img = Image.new("RGB", (page_w, total_h), BG)
    draw = ImageDraw.Draw(img)

    def draw_row(y: int, cells: list[str], height: int, f: ImageFont.FreeTypeFont, bold: bool = False):
        for ci, cell in enumerate(cells[:n_cols]):
            cx = MARGIN + ci * col_w
            draw.rectangle([cx, y, cx + col_w, y + height], outline=RULE)
            draw.text((cx + 4, y + 4), cell[:30], font=f, fill=FG)

    # Header
    draw_row(MARGIN, headers, header_h, hfont, bold=True)
    draw.rectangle(
        [MARGIN, MARGIN, MARGIN + n_cols * col_w, MARGIN + header_h],
        outline=FG
    )

    # Data rows
    for ri, row in enumerate(rows):
        y = MARGIN + header_h + ri * row_h
        draw_row(y, row, row_h, font)

    return img


# ---------------------------------------------------------------------------
# T3: Tanglish (Tamil + English code-mix)
# ---------------------------------------------------------------------------

def render_tanglish(
    lines: list[str],
    tamil_font_path: Path | str,
    latin_font_path: Path | str,
    font_size: int = FONT_SIZE,
    page_w: int = PAGE_W,
) -> Image.Image:
    """
    Render lines that mix Tamil script and Latin characters.

    Segments each line into Tamil/non-Tamil runs and renders each run
    with the appropriate font, advancing x position.
    """
    tfont = _load_font(tamil_font_path, font_size)
    lfont = _load_font(latin_font_path, font_size)
    lh = max(_line_height(tfont), _line_height(lfont)) + 4

    page_h = MARGIN * 2 + lh * max(len(lines), 1)
    img = Image.new("RGB", (page_w, page_h), BG)
    draw = ImageDraw.Draw(img)

    for li, line in enumerate(lines):
        y = MARGIN + li * lh
        x = MARGIN
        runs = _split_tanglish(line)
        for script, run_text in runs:
            font = tfont if script == "tamil" else lfont
            draw.text((x, y), run_text, font=font, fill=FG)
            bbox = font.getbbox(run_text)
            x += (bbox[2] - bbox[0]) + 2

    return img


def _split_tanglish(text: str) -> list[tuple[str, str]]:
    """Split text into (script, run) pairs: 'tamil' or 'latin'."""
    runs: list[tuple[str, str]] = []
    current_script = None
    current_run = ""

    for ch in text:
        cp = ord(ch)
        # Tamil Unicode block: 0x0B80–0x0BFF
        script = "tamil" if 0x0B80 <= cp <= 0x0BFF else "latin"
        if script == current_script:
            current_run += ch
        else:
            if current_run:
                runs.append((current_script, current_run))
            current_script = script
            current_run = ch

    if current_run:
        runs.append((current_script, current_run))

    return runs


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    fonts_dir = Path("fonts")
    tfont = fonts_dir / "NotoSansTamil-Regular.ttf"
    lfont = fonts_dir / "NotoSans-Regular.ttf"

    if not tfont.exists():
        print("Fonts not found — run download_fonts.py first")
        sys.exit(1)

    # T1
    lines = [
        "தமிழ் மொழி உலகின் மிகப் பழமையான மொழிகளில் ஒன்று.",
        "இது 2000 ஆண்டுகளுக்கும் மேலான இலக்கிய வரலாற்றைக் கொண்டுள்ளது.",
        "தமிழ் நாட்டில் சுமார் 7 கோடி மக்கள் பேசுகின்றனர்.",
    ]
    img = render_multicolumn(lines, tfont, columns=2)
    img.save("test_multicolumn.png")
    print(f"T1 multicolumn: {img.size} saved to test_multicolumn.png")

    # T2
    headers = ["வரிசை", "சொல்", "எழுத்துக்கள்"]
    rows = [["1", "தமிழ்", "5"], ["2", "மொழி", "4"], ["3", "இலக்கியம்", "9"]]
    img2 = render_table(rows, headers, tfont)
    img2.save("test_table.png")
    print(f"T2 table: {img2.size} saved to test_table.png")

    # T3
    mixed = ["Tamil என்பது ஒரு ancient language with rich history."]
    img3 = render_tanglish(mixed, tfont, lfont)
    img3.save("test_tanglish.png")
    print(f"T3 tanglish: {img3.size} saved to test_tanglish.png")

    print("synth_layout.py smoke test passed.")
