"""
infer_firered.py — inference wrapper for FireRedTeam/FireRed-OCR (2B).

FireRed-OCR (Feb 2026) is Qwen3-VL based with 3-stage progressive training
+ GRPO RL with task rewards.  Uses identical API to infer_qwen.py since the
base architecture is the same.

HF slug: FireRedTeam/FireRed-OCR
Outputs: results/firered/<manifest_stem>.jsonl
Each record: {id, script, mode, ground_truth, prediction, model}

Note: FireRed-OCR may have its own system prompt conventions; the standard
Qwen3-VL chat template is used here with an OCR-focused user message.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = Path("results") / "firered"
MODEL_ID = "FireRedTeam/FireRed-OCR"
OCR_PROMPT = "Please transcribe all text visible in this image accurately."


def run(manifest_path: str, max_samples: int | None = None) -> None:
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"FireRed-OCR inference on {device}", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

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
            img_uri = Path(rec["image_path"]).resolve().as_uri()

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_uri},
                        {"type": "text", "text": OCR_PROMPT},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=512)

            input_len = inputs["input_ids"].shape[1]
            pred = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)

            row = {
                "id": rec["id"],
                "script": rec.get("script", ""),
                "mode": rec.get("mode", ""),
                "ground_truth": rec["ground_truth"],
                "prediction": pred.strip(),
                "model": MODEL_ID,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(records)}", flush=True)

    print(f"Done. Results → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="FireRed-OCR inference")
    ap.add_argument("manifest", help="JSONL manifest from datagen.py")
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()
    run(args.manifest, args.max_samples)


if __name__ == "__main__":
    main()
