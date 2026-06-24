"""
infer_dotsocr.py — inference wrapper for rednote-hilab/dots.ocr (1.7B).

dots.ocr (rebranded from dots.mocr, March 2026) is a compact OCR-VLM from
rednote-hilab.  It is Qwen3-VL based (like FireRed-OCR) so we use the same
apply_chat_template pattern but with the dots.ocr model slug.

HF slug: rednote-hilab/dots.ocr
Outputs: results/dotsocr/<manifest_stem>.jsonl
Each record: {id, script, mode, ground_truth, prediction, model}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = Path("results") / "dotsocr"
MODEL_ID = "rednote-hilab/dots.ocr"
OCR_PROMPT = "Read the text in this image exactly, preserving all characters."


def run(manifest_path: str, max_samples: int | None = None) -> None:
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dots.ocr inference on {device}", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
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
            img = Image.open(rec["image_path"]).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": OCR_PROMPT},
                    ],
                }
            ]

            try:
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(device)
            except Exception:
                # Fallback
                inputs = processor(images=img, text=OCR_PROMPT, return_tensors="pt").to(device)

            # Filter out keys the model.generate() doesn't accept
            GENERATE_KEYS = {"input_ids", "attention_mask", "pixel_values",
                             "image_grid_thw", "position_ids"}
            gen_inputs = {k: v for k, v in inputs.items() if k in GENERATE_KEYS}
            # dots.ocr requires cache_position to be set explicitly
            seq_len = gen_inputs["input_ids"].shape[1]
            gen_inputs["cache_position"] = torch.arange(seq_len, device=gen_inputs["input_ids"].device)
            with torch.no_grad():
                output_ids = model.generate(**gen_inputs, max_new_tokens=512)

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
    ap = argparse.ArgumentParser(description="dots.ocr inference")
    ap.add_argument("manifest", help="JSONL manifest from datagen.py")
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()
    run(args.manifest, args.max_samples)


if __name__ == "__main__":
    main()
