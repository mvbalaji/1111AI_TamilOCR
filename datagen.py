"""
datagen.py — build dataset images + manifest for Gate A / Gate B / benchmark.

Outputs (under --out_dir):
  images/<split>/<script>/<real|scrambled>/<idx>.png
  manifests/<split>.jsonl

Default counts: Tamil=8000, Devanagari=1000, Latin=1000
Use --skip_scripts to skip scripts already generated (e.g. tamil).
Skipped scripts are loaded from --merge_manifest and appended to the new manifest.

Usage:
  # Full generation (all scripts)
  python datagen.py --out_dir data_train_v2 --use_corpora --splits train --multi_font --aug_level medium

  # Skip Tamil (already done), generate only Devanagari + Latin, merge with existing Tamil manifest
  python datagen.py --out_dir data_train_v2 --use_corpora --splits train --multi_font --aug_level medium \\
      --skip_scripts tamil \\
      --merge_manifest data_train/manifests/train.jsonl

  # Custom counts
  python datagen.py --out_dir data --n_tamil 8000 --n_devanagari 1000 --n_latin 1000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from textkit import normalize, scramble, segment, render


# ---------------------------------------------------------------------------
# Built-in fallback corpus
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
        # Confusable-heavy sentences
        "நண்பன் நாடகம் நன்றாக நடத்தினான் நண்பா",
        "ரதம் ரவி றவி றாகம் ரயில் வருகிறது",
        "ளவு ழவு லவு மூன்றும் மாறி வருகின்றன",
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

# Tanglish (Tamil+English code-mix) built-in lines
BUILTIN_TANGLISH = [
    "Tamil என்பது ஒரு ancient language with rich history.",
    "இந்த project-ல் OCR accuracy மிக முக்கியம்.",
    "Machine learning models Tamil text-ஐ படிக்க வேண்டும்.",
    "Research paper submission deadline நாளை.",
    "Dataset quality மிக நல்லா இருக்கு, let us proceed.",
]

FONT_MAP = {
    "tamil":      "fonts/NotoSansTamil-Regular.ttf",
    "devanagari": "fonts/NotoSansDevanagari-Regular.ttf",
    "latin":      "fonts/NotoSans-Regular.ttf",
}

# Additional Tamil font variants for multi-font diversity
TAMIL_FONT_VARIANTS = [
    "fonts/NotoSansTamil-Regular.ttf",
    "fonts/Catamaran-Regular.ttf",
    "fonts/TiroTamil-Regular.ttf",
    "fonts/HindMadurai-Regular.ttf",
    "fonts/BalooThambi2-Regular.ttf",
    "fonts/Arima-Regular.ttf",
    "fonts/MeeraInimai-Regular.ttf",
    "fonts/NotoSerifTamil-Regular.ttf",
]


def load_corpus(script: str, corpus_dir: Path, n: int, seed: int) -> list[str]:
    path = corpus_dir / f"{script}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rng = random.Random(seed)
    if len(lines) < n:
        lines = lines * (n // len(lines) + 1)
    return rng.sample(lines, n)


def _pick_tamil_font(idx: int, multi_font: bool) -> str:
    """Pick Tamil font: cycle through variants if multi_font, else use base."""
    if not multi_font:
        return FONT_MAP["tamil"]
    available = [f for f in TAMIL_FONT_VARIANTS if Path(f).exists()]
    if not available:
        return FONT_MAP["tamil"]
    return available[idx % len(available)]


def build(
    out_dir: Path,
    n_lines: int,
    seed: int,
    font_size: int,
    use_corpora: bool,
    corpus_dir: Path,
    splits: list[str] | None = None,
    multi_font: bool = False,
    aug_level: str = "clean",
    layout: str = "line",
    oversample_confusable: bool = False,
    skip_scripts: list[str] | None = None,
    merge_manifest: Path | None = None,
    n_per_script: dict[str, int] | None = None,
) -> None:
    if splits is None:
        splits = ["gate"]
    if skip_scripts is None:
        skip_scripts = []
    # Default per-script counts: Tamil=8000, Devanagari=1000, Latin=1000
    counts = {"tamil": 8000, "devanagari": 1000, "latin": 1000}
    if n_per_script:
        counts.update(n_per_script)

    # Import augmentation and layout modules lazily
    augment_fn = None
    if aug_level != "clean":
        try:
            from augment import augment_image
            augment_fn = augment_image
        except ImportError:
            print("WARN: augment.py not found — skipping augmentation")

    layout_renderers = {}
    if layout != "line":
        try:
            from synth_layout import render_multicolumn, render_table, render_tanglish
            layout_renderers = {
                "multicolumn": render_multicolumn,
                "table": render_table,
                "tanglish": render_tanglish,
            }
        except ImportError:
            print("WARN: synth_layout.py not found — falling back to line layout")
            layout = "line"

    root = out_dir
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        records: list[dict] = []

        # Handle Tanglish layout as a special pass-through
        if layout == "tanglish":
            _build_tanglish(
                records, root, split, n_lines, seed, font_size,
                use_corpora, corpus_dir, augment_fn, aug_level,
                layout_renderers.get("tanglish"),
            )
        else:
            for script, font_rel in FONT_MAP.items():
                if script in skip_scripts:
                    print(f"  [SKIP] {script} — will merge from existing manifest")
                    continue

                font_path = Path(font_rel)
                if not font_path.exists():
                    raise FileNotFoundError(
                        f"Missing font: {font_path}\n"
                        "Run: python download_fonts.py"
                    )

                n_script = counts.get(script, n_lines)
                if use_corpora:
                    lines = load_corpus(script, corpus_dir, n_script, seed)
                else:
                    src = BUILTIN[script]
                    lines = [normalize(src[i % len(src)]) for i in range(n_script)]

                if oversample_confusable and script == "tamil":
                    from benchmark_spec import CONFUSABLE_SETS
                    confusable_lines = _make_confusable_lines(CONFUSABLE_SETS, n_script // 5)
                    lines = lines[: n_script - len(confusable_lines)] + confusable_lines

                for mode in ("real", "scrambled"):
                    img_dir = root / "images" / split / script / mode
                    img_dir.mkdir(parents=True, exist_ok=True)

                    for idx, line in enumerate(lines):
                        text = line if mode == "real" else scramble(line, seed=seed + idx)
                        fp = _pick_tamil_font(idx, multi_font) if script == "tamil" else font_rel
                        font_path_i = Path(fp) if Path(fp).exists() else Path(font_rel)

                        try:
                            img = render(text, font_path_i, font_size=font_size)
                        except Exception as exc:
                            print(f"  WARN render [{script}/{mode}/{idx}]: {exc}")
                            continue

                        if augment_fn is not None:
                            img = augment_fn(img, seed=seed + idx, level=aug_level)

                        img_path = img_dir / f"{idx:05d}.png"
                        img.save(str(img_path))
                        if not img_path.exists():
                            print(f"  WARN save failed [{script}/{mode}/{idx}] — skipping")
                            continue

                        font_used = font_path_i.name if multi_font else font_rel
                        records.append({
                            "id": f"{split}_{script}_{mode}_{idx:05d}",
                            "split": split,
                            "script": script,
                            "mode": mode,
                            "text": text,
                            "ground_truth": line,
                            "image_path": str(img_path),
                            "grapheme_count": len(segment(text)),
                            "font": font_used,
                            "aug_level": aug_level,
                            "layout": layout,
                        })

                    print(f"  [{split}/{script}/{mode}] {len(lines)} images → {img_dir}")

            # Merge skipped scripts from an existing manifest
            if skip_scripts and merge_manifest and merge_manifest.exists():
                merged = 0
                with open(merge_manifest, encoding="utf-8") as mf:
                    for line in mf:
                        rec = json.loads(line)
                        if rec.get("script") in skip_scripts and rec.get("split") == split:
                            records.append(rec)
                            merged += 1
                print(f"  [MERGE] {merged} records from {merge_manifest} "
                      f"(scripts: {skip_scripts})")

        manifest_path = manifest_dir / f"{split}.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Manifest: {manifest_path}  ({len(records)} records)")


def _build_tanglish(
    records, root, split, n_lines, seed, font_size,
    use_corpora, corpus_dir, augment_fn, aug_level, render_fn,
):
    """Build Tanglish (code-mix) layout images."""
    tfont = Path(FONT_MAP["tamil"])
    lfont = Path(FONT_MAP["latin"])
    if not tfont.exists() or not lfont.exists():
        print("WARN: Tamil/Latin fonts missing for Tanglish — skipping")
        return

    if use_corpora and (corpus_dir / "tanglish.txt").exists():
        raw = load_corpus("tanglish", corpus_dir, n_lines, seed)
    else:
        raw = [normalize(BUILTIN_TANGLISH[i % len(BUILTIN_TANGLISH)]) for i in range(n_lines)]

    img_dir = root / "images" / split / "tanglish" / "real"
    img_dir.mkdir(parents=True, exist_ok=True)

    for idx, line in enumerate(raw):
        if render_fn is not None:
            img = render_fn([line], tfont, lfont, font_size=font_size)
        else:
            from textkit import render as line_render
            img = line_render(line, tfont, font_size=font_size)

        if augment_fn is not None:
            img = augment_fn(img, seed=seed + idx, level=aug_level)

        img_path = img_dir / f"{idx:05d}.png"
        img.save(str(img_path))
        records.append({
            "id": f"{split}_tanglish_real_{idx:05d}",
            "split": split,
            "script": "tanglish",
            "mode": "real",
            "text": line,
            "ground_truth": line,
            "image_path": str(img_path),
            "grapheme_count": len(segment(line)),
            "font": "mixed",
            "aug_level": aug_level,
            "layout": "tanglish",
        })
    print(f"  [{split}/tanglish/real] {len(raw)} images → {img_dir}")


def _make_confusable_lines(confusable_sets, n: int) -> list[str]:
    """Generate lines containing confusable grapheme pairs."""
    import unicodedata
    lines = []
    for group in confusable_sets:
        for char in group:
            lines.append(f"இது {char} எழுத்து உள்ள வரிசை {char}{char}{char}")
    return [normalize(l) for l in lines[:n]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",               default="data")
    ap.add_argument("--corpus_dir",            default="corpora")
    ap.add_argument("--n_lines",               type=int, default=200,
                    help="fallback line count if per-script count not set")
    ap.add_argument("--n_tamil",               type=int, default=8000,
                    help="Tamil image count (default: 8000)")
    ap.add_argument("--n_devanagari",          type=int, default=1000,
                    help="Devanagari image count (default: 1000)")
    ap.add_argument("--n_latin",               type=int, default=1000,
                    help="Latin image count (default: 1000)")
    ap.add_argument("--seed",                  type=int, default=42)
    ap.add_argument("--font_size",             type=int, default=32)
    ap.add_argument("--use_corpora",           action="store_true")
    ap.add_argument("--splits",                nargs="+", default=["gate"])
    ap.add_argument("--multi_font",            action="store_true",
                    help="cycle through all available Tamil fonts")
    ap.add_argument("--aug_level",             choices=["clean","light","medium","heavy"],
                    default="clean")
    ap.add_argument("--layout",                choices=["line","multicolumn","table","tanglish"],
                    default="line")
    ap.add_argument("--oversample_confusable", action="store_true",
                    help="add confusable-grapheme sentences to Tamil set")
    ap.add_argument("--skip_scripts",          nargs="+", default=[],
                    help="scripts to skip generation for (e.g. --skip_scripts tamil)")
    ap.add_argument("--merge_manifest",        default=None,
                    help="existing manifest to pull skipped-script records from")
    args = ap.parse_args()

    build(
        out_dir=Path(args.out_dir),
        n_lines=args.n_lines,
        seed=args.seed,
        font_size=args.font_size,
        use_corpora=args.use_corpora,
        corpus_dir=Path(args.corpus_dir),
        splits=args.splits,
        multi_font=args.multi_font,
        aug_level=args.aug_level,
        layout=args.layout,
        oversample_confusable=args.oversample_confusable,
        skip_scripts=args.skip_scripts,
        merge_manifest=Path(args.merge_manifest) if args.merge_manifest else None,
        n_per_script={
            "tamil":      args.n_tamil,
            "devanagari": args.n_devanagari,
            "latin":      args.n_latin,
        },
    )


if __name__ == "__main__":
    main()
