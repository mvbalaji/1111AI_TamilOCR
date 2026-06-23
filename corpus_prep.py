"""
corpus_prep.py — download and prepare corpora/<script>.txt for Gate A/B.

Writes one sentence per line, UTF-8, 20-200 chars, NFC normalized.
Target: ≥500 lines per script (datagen.py samples from these; Gate B needs ≥150).

Sources used (all permissive, no trust_remote_code required):
  Tamil      — wikimedia/wikipedia '20231101.ta'  (CC BY-SA 4.0)
               fallback: cc100 'ta'               (Common Crawl)
  Devanagari — wikimedia/wikipedia '20231101.hi'  (CC BY-SA 4.0)
               fallback: cc100 'hi'
  Latin      — wikimedia/wikipedia '20231101.en'  (CC BY-SA 4.0)

Note: ai4bharat/IndicCorp was removed from the Hub (June 2025).

Usage:
  pip install datasets
  python corpus_prep.py                  # all three scripts, 500 lines each
  python corpus_prep.py --langs tamil    # Tamil only
  python corpus_prep.py --n 1000        # 1000 lines per script
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Language configs
# ---------------------------------------------------------------------------

LANG_CONFIG = {
    "tamil": {
        "hf_dataset":  "wikimedia/wikipedia",
        "hf_config":   "20231101.ta",
        "hf_split":    "train",
        "text_field":  "text",
        "out_file":    "corpora/tamil.txt",
        "script_range": (0x0B80, 0x0BFF),   # Tamil Unicode block
        "fallback": ("cc100", "ta"),
    },
    "devanagari": {
        "hf_dataset":  "wikimedia/wikipedia",
        "hf_config":   "20231101.hi",
        "hf_split":    "train",
        "text_field":  "text",
        "out_file":    "corpora/devanagari.txt",
        "script_range": (0x0900, 0x097F),   # Devanagari Unicode block
        "fallback": ("cc100", "hi"),
    },
    "latin": {
        "hf_dataset":  "wikimedia/wikipedia",
        "hf_config":   "20231101.en",
        "hf_split":    "train",
        "text_field":  "text",
        "out_file":    "corpora/latin.txt",
        "script_range": None,               # ASCII range check inline
        "fallback": None,
    },
}

MIN_LEN = 20
MAX_LEN = 200
OVERSAMPLE_FACTOR = 8  # stream more than needed to fill quota after filtering


def _is_mostly_script(text: str, script_range: tuple[int, int] | None, threshold: float = 0.5) -> bool:
    """Check that ≥threshold fraction of non-space chars are in the target script block."""
    if script_range is None:
        # Latin: check ASCII printable fraction
        chars = [c for c in text if not c.isspace()]
        if not chars:
            return False
        latin = sum(1 for c in chars if ord(c) < 0x0250)
        return latin / len(chars) >= threshold
    lo, hi = script_range
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    in_block = sum(1 for c in chars if lo <= ord(c) <= hi)
    return in_block / len(chars) >= threshold


def _clean_line(line: str) -> str | None:
    """NFC normalize, strip, length-filter. Returns None if line is unusable."""
    line = unicodedata.normalize("NFC", line.strip())
    # Remove lines with excessive punctuation or numbers-only
    alnum = [c for c in line if c.isalpha()]
    if len(alnum) < 10:
        return None
    if not (MIN_LEN <= len(line) <= MAX_LEN):
        return None
    return line


def _extract_sentences(text: str) -> list[str]:
    """Split article/paragraph text into sentence-like lines."""
    import re
    # Split on sentence boundaries; keep moderate-length fragments
    parts = re.split(r'[।\.\!\?\n]+', text)
    return [p.strip() for p in parts if p.strip()]


def _load_streaming(dataset_id: str, config: str, split: str):
    """Load a streaming dataset — no trust_remote_code."""
    from datasets import load_dataset
    return load_dataset(dataset_id, config, split=split, streaming=True)


def download_corpus(lang: str, n: int, out_file: Path) -> int:
    """Stream from HF, extract and write n clean lines. Returns actual count written."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("datasets required: pip install datasets") from e

    cfg = LANG_CONFIG[lang]
    primary = (cfg["hf_dataset"], cfg["hf_config"])
    fallback = cfg.get("fallback")

    # Try primary source, fall back if it errors
    ds = None
    for attempt, (ds_id, ds_cfg) in enumerate([primary] + ([fallback] if fallback else [])):
        try:
            print(f"Streaming {ds_id} ({ds_cfg}) → {out_file}")
            ds = _load_streaming(ds_id, ds_cfg, cfg["hf_split"])
            break
        except Exception as e:
            if attempt == 0 and fallback:
                print(f"  Primary source failed ({e}), trying fallback {fallback[0]} ({fallback[1]}) ...")
            else:
                raise RuntimeError(
                    f"All sources failed for {lang}.\n"
                    f"Last error: {e}\n"
                    f"Try: python corpus_prep.py --langs {lang} --source manual"
                ) from e

    script_range = cfg["script_range"]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    examined = 0

    with open(out_file, "w", encoding="utf-8") as f:
        for example in ds:
            if written >= n:
                break
            raw = example.get(cfg["text_field"], "") or example.get("sentence", "")
            if not raw:
                continue
            for sent in _extract_sentences(raw):
                if written >= n:
                    break
                examined += 1
                if not _is_mostly_script(sent, script_range):
                    continue
                clean = _clean_line(sent)
                if clean is None:
                    continue
                f.write(clean + "\n")
                written += 1

    print(f"  {lang}: {written}/{n} lines written  (examined {examined} candidates)")
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=list(LANG_CONFIG.keys()),
                    choices=list(LANG_CONFIG.keys()))
    ap.add_argument("--n", type=int, default=500,
                    help="lines per script to collect (≥150 needed for Gate B)")
    args = ap.parse_args()

    for lang in args.langs:
        out = Path(LANG_CONFIG[lang]["out_file"])
        if out.exists():
            existing = sum(1 for _ in open(out, encoding="utf-8"))
            if existing >= args.n:
                print(f"  {lang}: {out} already has {existing} lines — skipping")
                continue
        count = download_corpus(lang, args.n, out)
        if count < 150:
            print(f"  WARNING: {lang} only got {count} lines — Gate B requires ≥150")

    print("\nDone. Run: python datagen.py --use_corpora")


if __name__ == "__main__":
    main()
