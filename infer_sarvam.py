"""
infer_sarvam.py — STUB for Sarvam AI OCR model.

WARNING: Sarvam AI's publication/redistribution policy for model weights is
not publicly confirmed as of June 2026.  This file is a stub only.
Do NOT use in benchmark results until:
  1. Verify model is publicly available on HuggingFace with an open license.
  2. Confirm Sarvam's terms allow benchmark result publication.
  3. Replace this stub with the verified API implementation.

Contact: https://www.sarvam.ai / check HuggingFace at sarvam-ai org.

Stub outputs empty predictions with a clear marker so any downstream eval
code can filter them out rather than silently computing wrong CERs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = Path("results") / "sarvam"
MODEL_ID = "sarvam-ai/PENDING_VERIFICATION"

WARNING = (
    "\n[SARVAM STUB] This wrapper is NOT implemented.\n"
    "Verify Sarvam AI publication policy before implementing.\n"
    "See infer_sarvam.py header for details.\n"
)


def run(manifest_path: str, max_samples: int | None = None) -> None:
    print(WARNING)

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
        for rec in records:
            row = {
                "id": rec["id"],
                "script": rec.get("script", ""),
                "mode": rec.get("mode", ""),
                "ground_truth": rec["ground_truth"],
                "prediction": "STUB_NOT_IMPLEMENTED",
                "model": MODEL_ID,
                "note": "Sarvam stub — policy unverified",
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Stub output written to {out_path}")
    print("All predictions are STUB_NOT_IMPLEMENTED — exclude from eval tables.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sarvam OCR stub (NOT implemented)")
    ap.add_argument("manifest", help="JSONL manifest from datagen.py")
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()
    run(args.manifest, args.max_samples)


if __name__ == "__main__":
    main()
