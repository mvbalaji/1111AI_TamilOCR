"""
textkit.py — grapheme segmentation, scrambling, rendering, ink-density.

Key design choices documented for Pillar 2:
- Segmentation uses \\X (Unicode grapheme cluster) via the `regex` package.
  Tamil akshara note: this correctly keeps கி (base+matira) and க் (base+pulli)
  as single clusters.  It splits க்ஷ into க்+ஷ  (akshara boundary, not a
  grapheme boundary).  We document this as an intentional akshara-vs-grapheme
  tradeoff: we measure grapheme clusters, not aksharas.  A future akshara-aware
  segmenter could be swapped in here.
- NFC normalization is applied before segmentation everywhere (Pillar 2 canonical
  form requirement).
- Scrambling destroys word order while preserving the exact grapheme multiset,
  giving a vision-only probe that removes the decoder language prior.
"""

from __future__ import annotations

import random
import unicodedata
from pathlib import Path
from typing import Sequence

import regex  # pip install regex  (not re — needs \\X support)

# ---------------------------------------------------------------------------
# Grapheme segmentation
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """NFC normalize — canonical Pillar-2 form."""
    return unicodedata.normalize("NFC", text)


def segment(text: str) -> list[str]:
    """Return list of Unicode grapheme clusters (\\X) after NFC normalization."""
    return regex.findall(r"\X", normalize(text))


def grapheme_count(text: str) -> int:
    return len(segment(text))


# ---------------------------------------------------------------------------
# Order-destroying scrambler (vision-only probe)
# ---------------------------------------------------------------------------

def scramble(text: str, seed: int = 42) -> str:
    """
    Destroy word order while preserving the exact grapheme multiset.
    All word boundaries are removed; grapheme clusters are shuffled globally.
    This removes the decoder language prior — any CER delta vs real text is
    attributable to visual decoding, not linguistic inference.
    """
    clusters = segment(text)
    rng = random.Random(seed)
    rng.shuffle(clusters)
    return "".join(clusters)


# ---------------------------------------------------------------------------
# Renderer — rasterize text to a PIL image
# ---------------------------------------------------------------------------

def render(
    text: str,
    font_path: str | Path,
    font_size: int = 32,
    padding: int = 8,
    bg: tuple[int, int, int] = (255, 255, 255),
    fg: tuple[int, int, int] = (0, 0, 0),
) -> "Image":  # type: ignore[name-defined]
    """Render text to a PIL RGB image at given font size."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:
        raise ImportError("Pillow required: pip install Pillow") from e

    font = ImageFont.truetype(str(font_path), font_size)

    # measure
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + 2 * padding
    h = bbox[3] - bbox[1] + 2 * padding

    img = Image.new("RGB", (max(w, 1), max(h, 1)), color=bg)
    draw = ImageDraw.Draw(img)
    draw.text((padding - bbox[0], padding - bbox[1]), text, font=font, fill=fg)
    return img


# ---------------------------------------------------------------------------
# Ink-density measurement
# ---------------------------------------------------------------------------

def ink_density(img: "Image") -> float:  # type: ignore[name-defined]
    """
    Fraction of pixels darker than threshold (ink pixels / total pixels).
    Used as the independent variable for Pillar 3 (glyph density per grapheme).
    """
    import numpy as np
    arr = np.array(img.convert("L"))
    ink = (arr < 128).sum()
    return float(ink) / arr.size


def ink_per_grapheme(text: str, font_path: str | Path, font_size: int = 32) -> float:
    """
    Render text then return (ink pixels) / (grapheme count).
    This is the density metric established in the empirical baseline:
      Latin ~117, Devanagari ~198, Tamil ~261  (pt32, Noto fonts).
    """
    img = render(text, font_path, font_size)
    n = grapheme_count(text)
    if n == 0:
        return 0.0
    import numpy as np
    arr = np.array(img.convert("L"))
    ink_pixels = int((arr < 128).sum())
    return ink_pixels / n


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    samples = {
        "tamil":      "தமிழ் நாடு",   # Tamil Nadu
        "devanagari": "नमस्ते दुनिया",
        "latin":      "Hello world",
    }
    for name, text in samples.items():
        clusters = segment(text)
        sc = scramble(text, seed=0)
        print(f"{name:>12}: {len(clusters):3d} clusters | '{text}' → scrambled '{sc}'")

    # Verify akshara-vs-grapheme documented choice
    tricky = "க்ஷ"
    segs = segment(tricky)
    print(f"\nக்ஷ segments into {len(segs)} clusters: {segs!r}  (documented: க்+ஷ split)")
