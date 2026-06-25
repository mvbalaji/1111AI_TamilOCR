"""
infer_gcloud.py — inference wrapper for Google Cloud Vision API (OCR).

Uses the DOCUMENT_TEXT_DETECTION feature for best accuracy on dense text.
Requires a GCP project with Vision API enabled.

Requirements:
  pip install google-cloud-vision

Setup (one of):
  # Option A — service account key file
  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

  # Option B — gcloud CLI
  gcloud auth application-default login

Usage:
  python infer_gcloud.py data/manifests/gate.jsonl
  python infer_gcloud.py data/manifests/gate.jsonl --max_samples 50

Outputs: results/gcloud/<manifest_stem>.jsonl
Each record: {id, script, mode, ground_truth, prediction, model, elapsed_s}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = Path("results") / "gcloud"
MODEL_ID    = "google-cloud-vision"


def infer_one(client, image_path: str) -> str:
    from google.cloud import vision

    with open(image_path, "rb") as f:
        content = f.read()

    image    = vision.Image(content=content)
    response = client.document_text_detection(image=image)

    if response.error.message:
        raise RuntimeError(f"GCloud Vision error: {response.error.message}")

    return response.full_text_annotation.text.strip()


def run(manifest_path: str, max_samples: int | None = None) -> None:
    try:
        from google.cloud import vision
    except ImportError:
        raise ImportError("pip install google-cloud-vision")

    client = vision.ImageAnnotatorClient()

    records = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            if r:
                records.append(r)

    if max_samples:
        records = records[:max_samples]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem     = Path(manifest_path).stem
    out_path = RESULTS_DIR / f"{stem}.jsonl"

    print(f"Google Cloud Vision — {len(records)} records → {out_path}", flush=True)

    with open(out_path, "w", encoding="utf-8") as out:
        for i, rec in enumerate(records):
            t0 = time.time()
            try:
                pred  = infer_one(client, rec["image_path"])
                error = None
            except Exception as exc:
                pred  = ""
                error = str(exc)
                print(f"  ERROR [{rec['id']}]: {exc}")

            row = {
                "id":           rec["id"],
                "script":       rec.get("script", ""),
                "mode":         rec.get("mode", ""),
                "ground_truth": rec["ground_truth"],
                "prediction":   pred,
                "model":        MODEL_ID,
                "elapsed_s":    round(time.time() - t0, 3),
            }
            if error:
                row["error"] = error
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(records)}", flush=True)

            # Stay within free-tier rate limit (1800 req/min)
            time.sleep(0.05)

    print(f"Done. Results → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Google Cloud Vision OCR inference")
    ap.add_argument("manifest",       help="JSONL manifest from datagen.py")
    ap.add_argument("--max_samples",  type=int, default=None)
    args = ap.parse_args()
    run(args.manifest, args.max_samples)


if __name__ == "__main__":
    main()
