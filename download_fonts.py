"""
download_fonts.py — fetch the three Noto OFL fonts needed by datagen.py.

Downloads variable-font TTFs from the google/fonts GitHub repo and saves them
as the canonical names datagen.py expects.  Safe to re-run (skips if present).

Usage:
    python download_fonts.py
    python download_fonts.py --force    # re-download even if files exist
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

FONTS_DIR = Path("fonts")

# (GitHub path under google/fonts main, local filename)
# Core fonts — used by datagen.py as canonical script representatives
FONT_DOWNLOADS = [
    (
        "ofl/notosanstamil/NotoSansTamil[wdth,wght].ttf",
        "NotoSansTamil-Regular.ttf",
    ),
    (
        "ofl/notosansdevanagari/NotoSansDevanagari[wdth,wght].ttf",
        "NotoSansDevanagari-Regular.ttf",
    ),
    (
        "ofl/notosans/NotoSans[wdth,wght].ttf",
        "NotoSans-Regular.ttf",
    ),
    # Additional Tamil fonts for multi-font P2 diversity
    (
        "ofl/catamaran/Catamaran[wght].ttf",
        "Catamaran-Regular.ttf",
    ),
    (
        "ofl/notoserif/NotoSerif[wdth,wght].ttf",
        "NotoSerifTamil-Regular.ttf",  # Tamil subset via variable font
    ),
    (
        "ofl/tirotamil/TiroTamil-Regular.ttf",
        "TiroTamil-Regular.ttf",
    ),
    (
        "ofl/hindmadurai/HindMadurai-Regular.ttf",
        "HindMadurai-Regular.ttf",
    ),
    (
        "ofl/baloothambi2/BalooThambi2[wght].ttf",
        "BalooThambi2-Regular.ttf",
    ),
    (
        "ofl/arima/Arima[wght].ttf",
        "Arima-Regular.ttf",
    ),
    (
        "ofl/meerainimai/MeeraInimai-Regular.ttf",
        "MeeraInimai-Regular.ttf",
    ),
]

# Canonical Tamil fonts for multi-font rendering in datagen.py
TAMIL_FONT_VARIANTS = [
    "NotoSansTamil-Regular.ttf",
    "Catamaran-Regular.ttf",
    "TiroTamil-Regular.ttf",
    "HindMadurai-Regular.ttf",
    "BalooThambi2-Regular.ttf",
    "Arima-Regular.ttf",
    "MeeraInimai-Regular.ttf",
    "NotoSerifTamil-Regular.ttf",
]

BASE_URL = "https://raw.githubusercontent.com/google/fonts/main/"


def download(force: bool = False) -> None:
    FONTS_DIR.mkdir(exist_ok=True)
    for src, dest in FONT_DOWNLOADS:
        out = FONTS_DIR / dest
        if out.exists() and not force:
            print(f"  {dest}: already present ({out.stat().st_size // 1024} KB) — skip")
            continue
        url = BASE_URL + src
        print(f"  {dest}: downloading ...", end=" ", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        urllib.request.urlretrieve(url, out)
        print(f"{out.stat().st_size // 1024} KB")

    print("\nVerifying renders ...")
    try:
        from textkit import render, segment
        checks = [
            ("தமிழ்",   "NotoSansTamil-Regular.ttf"),
            ("हिंदी",   "NotoSansDevanagari-Regular.ttf"),
            ("Hello",   "NotoSans-Regular.ttf"),
        ]
        for text, fname in checks:
            img = render(text, FONTS_DIR / fname, font_size=32)
            clusters = len(segment(text))
            print(f"  {fname}: rendered '{text}' → {img.size[0]}×{img.size[1]}px  "
                  f"({clusters} clusters) OK")
    except ImportError:
        print("  (Pillow/regex not installed — skipping render check)")

    print("\nFonts ready. Run: python validate_harness.py")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()
    download(force=args.force)


if __name__ == "__main__":
    main()
