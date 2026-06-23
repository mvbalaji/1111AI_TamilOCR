"""
infer_qwen.py — Qwen3-VL zero-shot inference (A100).

VERIFIED API (Qwen/Qwen3-VL-2B-Instruct):
  Class:    Qwen3VLForConditionalGeneration  ✓
  Loading:  AutoProcessor + model.from_pretrained  ✓
  Inputs:   processor.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt")
            Images passed as local file:// URIs or paths inside the messages dict.

  IMPORTANT: transformers 4.57.0 is not yet released — install from source:
    pip install git+https://github.com/huggingface/transformers
  qwen-vl-utils is NOT needed for Qwen3-VL (it was required for Qwen2-VL's
  process_vision_info; Qwen3-VL's processor handles images natively).

  Resolution/token control: pass min_pixels and max_pixels to AutoProcessor.
    Default: min_pixels=256*28*28, max_pixels=1280*28*28
    For a tighter budget (faster, fewer tokens): max_pixels=512*28*28

  Source: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct

Usage:
  python infer_qwen.py \\
      --manifest data/manifests/gate.jsonl \\
      --out results/gate_b_qwen.jsonl \\
      --mode real
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

QWEN_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

PROMPT = (
    "Transcribe the text in this image exactly as it appears. "
    "Output only the transcribed text, nothing else."
)

# Processor pixel bounds — controls visual token budget.
# Default (None) uses model-card defaults: min=256*28*28, max=1280*28*28.
DEFAULT_MIN_PIXELS = 256 * 28 * 28
DEFAULT_MAX_PIXELS = 1280 * 28 * 28


def load_model(model_id: str, device: str = "cuda", max_pixels: int = DEFAULT_MAX_PIXELS):
    """Load Qwen3-VL model + processor."""
    import torch
    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    except ImportError as e:
        raise ImportError(
            "Qwen3VLForConditionalGeneration not found.\n"
            "Install transformers from source:\n"
            "  pip install git+https://github.com/huggingface/transformers"
        ) from e

    print(f"Loading processor: {model_id}  (max_pixels={max_pixels})")
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=DEFAULT_MIN_PIXELS,
        max_pixels=max_pixels,
    )

    print(f"Loading model: {model_id}  (bfloat16)")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    return model, processor


def infer_one(model, processor, image_path: str, device: str = "cuda") -> str:
    """
    Run OCR on one image.
    Images are passed as file:// URIs so the processor loads them directly
    without needing qwen-vl-utils.
    """
    import torch

    # Qwen3-VL processor accepts local paths or file:// URIs.
    img_uri = Path(image_path).resolve().as_uri()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img_uri},
            {"type": "text",  "text": PROMPT},
        ],
    }]

    # apply_chat_template with tokenize=True returns a BatchFeature directly.
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    # Strip the input prompt tokens from the generated output.
    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[0, input_len:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def run_inference(
    manifest_path: Path,
    out_path: Path,
    mode_filter: str | None,
    script_filter: list[str] | None,
    device: str = "cuda",
    model_id: str = QWEN_MODEL_ID,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> None:
    model, processor = load_model(model_id, device, max_pixels)

    records: list[dict] = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if mode_filter and rec.get("mode") != mode_filter:
                continue
            if script_filter and rec.get("script") not in script_filter:
                continue
            records.append(rec)

    print(f"Qwen3-VL inference: {len(records)} records")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, rec in enumerate(records):
            t0 = time.time()
            try:
                pred = infer_one(model, processor, rec["image_path"], device)
                error = None
            except Exception as exc:
                pred = ""
                error = str(exc)
                print(f"  WARN [{rec['id']}]: {exc}")

            elapsed = time.time() - t0
            row = {
                "id":             rec["id"],
                "script":         rec.get("script"),
                "mode":           rec.get("mode"),
                "prediction":     pred,
                "ground_truth":   rec.get("ground_truth"),
                "grapheme_count": rec.get("grapheme_count"),
                "elapsed_s":      round(elapsed, 3),
                "model":          model_id,
            }
            if error:
                row["error"] = error
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(records)} done")

    print(f"Results → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",   required=True)
    ap.add_argument("--out",        required=True)
    ap.add_argument("--mode",       default=None, choices=["real", "scrambled"])
    ap.add_argument("--script",     nargs="+", default=None)
    ap.add_argument("--device",     default="cuda")
    ap.add_argument("--model_id",   default=QWEN_MODEL_ID)
    ap.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS,
                    help="max image pixels for processor (controls visual token budget)")
    args = ap.parse_args()

    run_inference(
        manifest_path=Path(args.manifest),
        out_path=Path(args.out),
        mode_filter=args.mode,
        script_filter=args.script,
        device=args.device,
        model_id=args.model_id,
        max_pixels=args.max_pixels,
    )


if __name__ == "__main__":
    main()
