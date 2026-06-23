"""
infer_tesseract.py — CPU-runnable Tesseract baseline for Tamil OCR benchmark.

Uses pytesseract (Python wrapper for Tesseract 5+).  Tamil language pack
must be installed: `apt-get install tesseract-ocr-tam` on Linux, or via
Tesseract installer with Tamil data files on Windows.

Language codes:
  Tamil     : tam
  Devanagari: hin  (Hindi)
  Latin     : eng

Outputs: results/tesseract/<manifest_stem>.jsonl
Each record: {id, script, mode, ground_truth, prediction, model, lang}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = Path("results") / "tesseract"
MODEL_ID = "tesseract-5"

SCRIPT_TO_LANG = {
    "tamil": "tam",
    "devanagari": "hin",
    "latin": "eng",
}


def _check_tesseract() -> bool:
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract {version} found", flush=True)
        return True
    except Exception as e:
        print(f"Tesseract not available: {e}", flush=True)
        return False


def run(manifest_path: str, max_samples: int | None = None) -> None:
    if not _check_tesseract():
        print(
            "Install: pip install pytesseract\n"
            "  Linux:   apt-get install tesseract-ocr tesseract-ocr-tam\n"
            "  Windows: https://github.com/UB-Mannheim/tesseract/wiki"
        )
        sys.exit(1)

    import pytesseract
    from PIL import Image

    records = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            if r:
                records.append(r)

    if max_samples:
        records = records[:max_samples]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(manifest_path).stem
    out_path = RESULTS_DIR / f"{stem}.jsonl"

    with open(out_path, "w", encoding="utf-8") as out:
        for i, rec in enumerate(records):
            img = Image.open(rec["image_path"]).convert("RGB")
            script = rec.get("script", "latin").lower()
            lang = SCRIPT_TO_LANG.get(script, "eng")

            try:
                pred = pytesseract.image_to_string(img, lang=lang, config="--psm 6")
                pred = pred.strip().replace("\n", " ")
            except pytesseract.TesseractError as e:
                pred = f"ERROR:{e}"

            row = {
                "id": rec["id"],
                "script": script,
                "mode": rec.get("mode", ""),
                "ground_truth": rec["ground_truth"],
                "prediction": pred,
                "model": MODEL_ID,
                "lang": lang,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(records)}", flush=True)

    print(f"Done. Results → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Tesseract 5 baseline inference")
    ap.add_argument("manifest", help="JSONL manifest from datagen.py")
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()
    run(args.manifest, args.max_samples)


if __name__ == "__main__":
    main()
