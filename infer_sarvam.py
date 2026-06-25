"""
infer_sarvam.py — inference wrapper for Sarvam Vision OCR API.

Sarvam AI provides a hosted vision API compatible with the OpenAI chat
completions format at api.sarvam.ai.

Requirements:
  pip install requests

Setup:
  export SARVAM_API_KEY=<your key from https://dashboard.sarvam.ai>

Usage:
  python infer_sarvam.py data/manifests/gate.jsonl
  python infer_sarvam.py data/manifests/gate.jsonl --max_samples 50

Outputs: results/sarvam/<manifest_stem>.jsonl
Each record: {id, script, mode, ground_truth, prediction, model, elapsed_s}
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = Path("results") / "sarvam"
# Sarvam multimodal model — check https://dashboard.sarvam.ai/docs for latest name
MODEL_ID    = "sarvam-m"
API_URL     = "https://api.sarvam.ai/v1/chat/completions"
OCR_PROMPT  = (
    "Transcribe the text in this image exactly as it appears. "
    "Output only the transcribed text, nothing else."
)


def encode_image(image_path: str) -> tuple[str, str]:
    """Returns (base64_data, mime_type)."""
    ext  = Path(image_path).suffix.lstrip(".").lower()
    mime = f"image/{ext}" if ext in ("png", "jpg", "jpeg", "webp") else "image/png"
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


def infer_one(api_key: str, image_path: str, retries: int = 3) -> str:
    import requests

    b64, mime = encode_image(image_path)
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type":      "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ],
        "max_tokens":  512,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    for attempt in range(retries):
        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
            if not resp.ok:
                # Print full error body to help diagnose API issues
                print(f"  API error {resp.status_code}: {resp.text[:500]}")
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  WARN attempt {attempt+1}: {exc} — retry in {wait}s")
                time.sleep(wait)
            else:
                raise


def run(manifest_path: str, max_samples: int | None = None) -> None:
    api_key = os.environ.get("SARVAM_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "SARVAM_API_KEY not set.\n"
            "Get your key from https://dashboard.sarvam.ai\n"
            "Then: export SARVAM_API_KEY=your_key"
        )

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

    print(f"Sarvam Vision — {len(records)} records → {out_path}", flush=True)

    with open(out_path, "w", encoding="utf-8") as out:
        for i, rec in enumerate(records):
            t0 = time.time()
            try:
                pred  = infer_one(api_key, rec["image_path"])
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

    print(f"Done. Results → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sarvam Vision OCR inference")
    ap.add_argument("manifest",       help="JSONL manifest from datagen.py")
    ap.add_argument("--max_samples",  type=int, default=None)
    args = ap.parse_args()
    run(args.manifest, args.max_samples)


if __name__ == "__main__":
    main()
