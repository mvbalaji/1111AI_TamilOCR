"""
datagen.py — build dataset images + manifest for Gate A / Gate B.

Outputs (under --out_dir):
  images/<split>/<script>/<real|scrambled>/<idx>.png
  manifests/<split>.jsonl   — {"id", "script", "mode", "text", "image_path", "grapheme_count"}

Usage:
  python datagen.py --out_dir data --n_lines 200 --seed 42
  python datagen.py --out_dir data --n_lines 200 --seed 42 --use_corpora

Corpus files (optional, --use_corpora):
  corpora/tamil.txt       one line of Tamil text per row
  corpora/devanagari.txt  one line of Devanagari text per row
  corpora/latin.txt       one line of Latin text per row

Font files (OFL, must be present):
  fonts/NotoSansTamil-Regular.ttf
  fonts/NotoSansDevanagari-Regular.ttf
  fonts/NotoSans-Regular.ttf

If font files are absent the script exits with a clear error message and font
download instructions.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from textkit import normalize, scramble, segment, render


# ---------------------------------------------------------------------------
# Built-in fallback corpus (short; used when --use_corpora not given)
# ---------------------------------------------------------------------------

BUILTIN: dict[str, list[str]] = {
    "tamil": [
        "தமிழ் மொழி உலகின் பழமையான மொழிகளில் ஒன்றாகும்",
        "வணக்கம் நண்பர்களே இன்று மகிழ்ச்சியான நாள்",
        "பள்ளி மாணவர்கள் தினமும் கல்வி கற்கின்றனர்",
        "இந்திய அரசியல் சாசனம் அனைவருக்கும் சம உரிமை அளிக்கிறது",
        "சென்னை தமிழ்நாட்டின் தலைநகரமாகும்",
        "கோவில் கோபுரம் மிகவும் உயரமாக இருக்கிறது",
        "மழை நீர் சேகரிப்பு முறை மிக முக்கியமானது",
        "குழந்தைகள் விளையாட்டு மைதானத்தில் ஓடுகின்றனர்",
        "அறிவியல் மற்றும் தொழில்நுட்பம் வளர்ந்து வருகிறது",
        "நூலகத்தில் ஆயிரக்கணக்கான புத்தகங்கள் உள்ளன",
    ],
    "devanagari": [
        "हिंदी भारत की राजभाषा है और इसे करोड़ों लोग बोलते हैं",
        "विज्ञान और प्रौद्योगिकी ने जीवन को सरल बनाया है",
        "भारतीय संविधान सभी नागरिकों को समान अधिकार देता है",
        "दिल्ली भारत की राजधानी है और यहाँ अनेक दर्शनीय स्थल हैं",
        "पुस्तकालय में हजारों पुस्तकें उपलब्ध हैं",
        "बच्चे प्रतिदिन विद्यालय जाते हैं और पढ़ाई करते हैं",
        "वर्षा जल संचयन एक महत्वपूर्ण विधि है",
        "मंदिर का शिखर बहुत ऊँचा है",
        "खेल के मैदान में बच्चे खेल रहे हैं",
        "भारतीय संस्कृति अत्यंत समृद्ध और विविध है",
    ],
    "latin": [
        "The quick brown fox jumps over the lazy dog",
        "Science and technology have transformed modern life",
        "Libraries contain thousands of books and resources",
        "Children go to school every day to learn new things",
        "Rainwater harvesting is an important conservation method",
        "The temple tower is very tall and ornate",
        "The constitution grants equal rights to all citizens",
        "Students study mathematics physics and literature",
        "The capital city has many historical monuments",
        "Education is the foundation of a prosperous society",
    ],
}

FONT_MAP = {
    "tamil":      "fonts/NotoSansTamil-Regular.ttf",
    "devanagari": "fonts/NotoSansDevanagari-Regular.ttf",
    "latin":      "fonts/NotoSans-Regular.ttf",
}

FONT_DOWNLOAD_MSG = """
Missing font file: {path}
Download OFL fonts from Google Fonts and place them at:
  fonts/NotoSansTamil-Regular.ttf
  fonts/NotoSansDevanagari-Regular.ttf
  fonts/NotoSans-Regular.ttf

Quick download (Linux/Mac):
  pip install gfonts
  gfonts download "Noto Sans Tamil" "Noto Sans Devanagari" "Noto Sans"
  # or download manually from https://fonts.google.com/noto
"""


def load_corpus(script: str, corpus_dir: Path, n: int, seed: int) -> list[str]:
    path = corpus_dir / f"{script}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rng = random.Random(seed)
    if len(lines) < n:
        lines = lines * (n // len(lines) + 1)
    return rng.sample(lines, n)


def build(
    out_dir: Path,
    n_lines: int,
    seed: int,
    font_size: int,
    use_corpora: bool,
    corpus_dir: Path,
    splits: list[str] | None = None,
) -> None:
    if splits is None:
        splits = ["gate"]  # single split for gate experiments

    root = out_dir
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        records: list[dict] = []
        for script, font_rel in FONT_MAP.items():
            font_path = Path(font_rel)
            if not font_path.exists():
                raise FileNotFoundError(FONT_DOWNLOAD_MSG.format(path=font_path))

            if use_corpora:
                lines = load_corpus(script, corpus_dir, n_lines, seed)
            else:
                src = BUILTIN[script]
                rng = random.Random(seed + hash(script))
                lines = [src[i % len(src)] for i in range(n_lines)]
                # add mild variation by appending index token
                lines = [normalize(ln) for ln in lines]

            for mode in ("real", "scrambled"):
                img_dir = root / "images" / split / script / mode
                img_dir.mkdir(parents=True, exist_ok=True)
                for idx, line in enumerate(lines):
                    text = line if mode == "real" else scramble(line, seed=seed + idx)
                    try:
                        img = render(text, font_path, font_size=font_size)
                    except Exception as exc:
                        print(f"  WARN render failed [{script}/{mode}/{idx}]: {exc}")
                        continue
                    img_path = img_dir / f"{idx:05d}.png"
                    img.save(str(img_path))
                    records.append({
                        "id": f"{split}_{script}_{mode}_{idx:05d}",
                        "split": split,
                        "script": script,
                        "mode": mode,
                        "text": text,
                        "ground_truth": line,
                        "image_path": str(img_path),
                        "grapheme_count": len(segment(text)),
                    })
                print(f"  [{split}/{script}/{mode}] {len(lines)} images written to {img_dir}")

        manifest_path = manifest_dir / f"{split}.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Manifest: {manifest_path}  ({len(records)} records)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",     default="data",   help="output root")
    ap.add_argument("--corpus_dir",  default="corpora")
    ap.add_argument("--n_lines",     type=int, default=200)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--font_size",   type=int, default=32)
    ap.add_argument("--use_corpora", action="store_true",
                    help="read from corpora/<script>.txt instead of built-ins")
    ap.add_argument("--splits",      nargs="+", default=["gate"])
    args = ap.parse_args()

    build(
        out_dir=Path(args.out_dir),
        n_lines=args.n_lines,
        seed=args.seed,
        font_size=args.font_size,
        use_corpora=args.use_corpora,
        corpus_dir=Path(args.corpus_dir),
        splits=args.splits,
    )


if __name__ == "__main__":
    main()
