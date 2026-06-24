"""
evaluate_ft.py — Evaluate the fine-tuned LoRA model on the gate benchmark.

Loads the LoRA adapter on top of Qwen3-VL-2B and runs inference on the
gate manifest, then computes grapheme-CER vs the Qwen3-VL baseline.

Usage:
  python evaluate_ft.py \\
      --manifest   data/manifests/gate.jsonl \\
      --checkpoint checkpoints/tamil-ocr-v1/adapter_final \\
      --out        results/ft_eval.jsonl

  # Compare vs baseline
  python evaluate_ft.py \\
      --manifest   data/manifests/gate.jsonl \\
      --checkpoint checkpoints/tamil-ocr-v1/adapter_final \\
      --baseline   results/gate_a_qwen.jsonl \\
      --out        results/ft_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evaluate import grapheme_cer, codepoint_cer

BASE_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OCR_PROMPT = (
    "Transcribe the text in this image exactly as it appears. "
    "Output only the transcribed text, nothing else."
)
SCRIPTS = ["tamil", "devanagari", "latin"]


def load_model(checkpoint_path: str, device: str = "cuda"):
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    print(f"Loading base model: {BASE_MODEL_ID}")
    processor = AutoProcessor.from_pretrained(
        checkpoint_path,
        min_pixels=256 * 28 * 28,
        max_pixels=1024 * 28 * 28,
    )
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    print(f"Loading LoRA adapter: {checkpoint_path}")
    model = PeftModel.from_pretrained(base, checkpoint_path)
    model.eval()
    return model, processor


def infer_one(model, processor, image_path: str, device: str = "cuda") -> str:
    import io
    import torch
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    img_clean = Image.open(buf)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img_clean},
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
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    input_len  = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[0, input_len:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def print_comparison(ft_scores: dict, baseline_scores: dict | None) -> None:
    print(f"\n{'='*65}")
    print(f"{'script':>12} | {'FT grapheme-CER':>16} | {'baseline':>10} | {'delta':>8}")
    print("  " + "-" * 55)
    for sc in SCRIPTS:
        ft  = ft_scores.get(sc, {}).get("grapheme_cer", float("nan"))
        bl  = baseline_scores.get(sc, float("nan")) if baseline_scores else float("nan")
        delta = ft - bl if baseline_scores else float("nan")
        delta_str = f"{delta:+.4f}" if baseline_scores else "N/A"
        print(f"  {sc:>12} | {ft:>16.4f} | {bl:>10.4f} | {delta_str:>8}")
    print(f"{'='*65}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",    required=True)
    ap.add_argument("--checkpoint",  required=True,
                    help="path to adapter_final/ from finetune.py")
    ap.add_argument("--baseline",    default=None,
                    help="gate_a_qwen.jsonl for before/after comparison")
    ap.add_argument("--out",         required=True)
    ap.add_argument("--mode",        default="real", choices=["real", "scrambled", "both"])
    ap.add_argument("--device",      default="cuda")
    args = ap.parse_args()

    model, processor = load_model(args.checkpoint, args.device)

    # Load manifest
    records = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if args.mode != "both" and rec.get("mode") != args.mode:
                continue
            records.append(rec)

    print(f"Evaluating {len(records)} records...")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    results = []
    with open(args.out, "w", encoding="utf-8") as out_f:
        for i, rec in enumerate(records):
            t0 = time.time()
            try:
                pred  = infer_one(model, processor, rec["image_path"], args.device)
                error = None
            except Exception as exc:
                pred  = ""
                error = str(exc)
                print(f"  WARN [{rec['id']}]: {exc}")

            gt    = rec.get("ground_truth", "")
            gcer  = grapheme_cer(pred, gt)
            ccer  = codepoint_cer(pred, gt)

            row = {
                "id":            rec["id"],
                "script":        rec.get("script"),
                "mode":          rec.get("mode"),
                "prediction":    pred,
                "ground_truth":  gt,
                "grapheme_cer":  round(gcer, 4),
                "codepoint_cer": round(ccer, 4),
                "elapsed_s":     round(time.time() - t0, 3),
                "model":         f"ft:{args.checkpoint}",
            }
            if error:
                row["error"] = error
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            results.append(row)

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(records)} done")

    # Summary by script
    ft_scores: dict[str, dict] = {}
    for sc in SCRIPTS:
        sc_rows = [r for r in results if r.get("script") == sc]
        if sc_rows:
            ft_scores[sc] = {
                "grapheme_cer":  round(_mean([r["grapheme_cer"]  for r in sc_rows]), 4),
                "codepoint_cer": round(_mean([r["codepoint_cer"] for r in sc_rows]), 4),
                "n": len(sc_rows),
            }

    # Load baseline for comparison
    baseline_scores: dict[str, float] | None = None
    if args.baseline and Path(args.baseline).exists():
        bl_by_script: dict[str, list[float]] = {}
        with open(args.baseline, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("mode") != args.mode:
                    continue
                sc = rec.get("script", "")
                bl_by_script.setdefault(sc, []).append(
                    grapheme_cer(rec.get("prediction",""), rec.get("ground_truth",""))
                )
        baseline_scores = {sc: _mean(v) for sc, v in bl_by_script.items()}

    print_comparison(ft_scores, baseline_scores)

    # Save summary JSON
    summary_path = Path(args.out).with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "mode":       args.mode,
            "scores":     ft_scores,
            "baseline":   baseline_scores,
        }, f, indent=2, ensure_ascii=False)
    print(f"Results  → {args.out}")
    print(f"Summary  → {summary_path}")


if __name__ == "__main__":
    main()
