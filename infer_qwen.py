"""
infer_qwen.py — Qwen3-VL zero-shot inference (A100).

VERIFIED API (Qwen/Qwen3-VL-2B-Instruct):
  Class:    Qwen3VLForConditionalGeneration
  Loading:  AutoProcessor + model.from_pretrained
  Inputs:   processor.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt")
  Images:   passed as file:// URIs inside the messages dict.

REQUIREMENTS:
  Install transformers from source (Qwen3-VL not in 4.46.x release):
    pip install git+https://github.com/huggingface/transformers

Budget tiers — mapped to max_pixels so visual token counts match DeepSeek:
  Budget  | max_pixels       | ~vision tokens
  --------|-----------------|---------------
  tiny    | 256 * 28 * 28   | ~256
  small   | 400 * 28 * 28   | ~400
  base    | 1024 * 28 * 28  | ~1024
  gundam  | 1280 * 28 * 28  | ~1280  (model-card default)

Usage:
  # Gate A — all budgets (scrambled + real)
  python infer_qwen.py \\
      --manifest data/manifests/gate.jsonl \\
      --out results/gate_a_qwen.jsonl \\
      --budget tiny small base gundam

  # Gate B — real text only, base budget
  python infer_qwen.py \\
      --manifest data/manifests/gate.jsonl \\
      --out results/gate_b_qwen.jsonl \\
      --mode real --budget base
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

QWEN_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

OCR_PROMPT = (
    "Transcribe the text in this image exactly as it appears. "
    "Output only the transcribed text, nothing else."
)

# Budget → max_pixels.  Each 28×28 patch = 1 visual token.
BUDGET_CONFIG: dict[str, dict] = {
    "tiny":   {"max_pixels": 256  * 28 * 28},
    "small":  {"max_pixels": 400  * 28 * 28},
    "base":   {"max_pixels": 1024 * 28 * 28},
    "gundam": {"max_pixels": 1280 * 28 * 28},
}

BUDGET_VTOKENS_APPROX: dict[str, int] = {
    "tiny":   256,
    "small":  400,
    "base":   1024,
    "gundam": 1280,
}

DEFAULT_MIN_PIXELS = 256 * 28 * 28


def load_model(model_id: str, device: str = "cuda", max_pixels: int = 1280 * 28 * 28):
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


def _make_processor(model_id: str, max_pixels: int):
    """Load a processor with a specific pixel budget (reuses cached weights)."""
    try:
        from transformers import AutoProcessor
    except ImportError:
        raise

    return AutoProcessor.from_pretrained(
        model_id,
        min_pixels=DEFAULT_MIN_PIXELS,
        max_pixels=max_pixels,
    )


def infer_one(model, processor, image_path: str, device: str = "cuda") -> str:
    import torch
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text": OCR_PROMPT},
        ],
    }]

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

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[0, input_len:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def run_inference(
    manifest_path: Path,
    out_path: Path,
    budgets: list[str],
    mode_filter: str | None,
    script_filter: list[str] | None,
    device: str = "cuda",
    model_id: str = QWEN_MODEL_ID,
) -> None:
    # Load model weights once at the largest budget's max_pixels
    max_px = max(BUDGET_CONFIG[b]["max_pixels"] for b in budgets)
    model, _ = load_model(model_id, device, max_px)

    records: list[dict] = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if mode_filter and rec.get("mode") != mode_filter:
                continue
            if script_filter and rec.get("script") not in script_filter:
                continue
            records.append(rec)

    print(f"Qwen3-VL inference: {len(records)} records × {len(budgets)} budgets "
          f"= {len(records) * len(budgets)} calls")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(records) * len(budgets)
    done = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for budget in budgets:
            cfg = BUDGET_CONFIG[budget]
            # Swap processor for this budget's pixel cap
            processor = _make_processor(model_id, cfg["max_pixels"])
            vtokens = BUDGET_VTOKENS_APPROX[budget]

            for rec in records:
                img_path = rec.get("image_path", "")
                if not img_path or not Path(img_path).exists():
                    print(f"  SKIP [{rec['id']}]: image missing at '{img_path}'")
                    done += 1
                    continue

                t0 = time.time()
                try:
                    pred = infer_one(model, processor, img_path, device)
                    error = None
                except Exception as exc:
                    pred = ""
                    error = str(exc)
                    print(f"  WARN [{rec['id']}][{budget}]: {exc}")

                elapsed = time.time() - t0
                row = {
                    "id":             rec["id"] + f"__{budget}",
                    "base_id":        rec["id"],
                    "script":         rec.get("script"),
                    "mode":           rec.get("mode"),
                    "budget":         budget,
                    "vision_tokens":  vtokens,
                    "grapheme_count": rec.get("grapheme_count"),
                    "prediction":     pred,
                    "ground_truth":   rec.get("ground_truth"),
                    "elapsed_s":      round(elapsed, 3),
                    "model":          model_id,
                }
                if error:
                    row["error"] = error
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                done += 1
                if done % 100 == 0:
                    print(f"  {done}/{total} done")

    print(f"Results → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",   required=True)
    ap.add_argument("--out",        required=True)
    ap.add_argument("--budget",     nargs="+", default=["tiny", "small", "base", "gundam"],
                    choices=list(BUDGET_CONFIG.keys()))
    ap.add_argument("--mode",       default=None, choices=["real", "scrambled"])
    ap.add_argument("--script",     nargs="+", default=None)
    ap.add_argument("--device",     default="cuda")
    ap.add_argument("--model_id",   default=QWEN_MODEL_ID)
    args = ap.parse_args()

    run_inference(
        manifest_path=Path(args.manifest),
        out_path=Path(args.out),
        budgets=args.budget,
        mode_filter=args.mode,
        script_filter=args.script,
        device=args.device,
        model_id=args.model_id,
    )


if __name__ == "__main__":
    main()
