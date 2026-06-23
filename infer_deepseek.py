"""
infer_deepseek.py — DeepSeek-OCR inference across compression budgets (A100).

VERIFIED API (deepseek-ai/DeepSeek-OCR, deepseek-ai/DeepSeek-OCR-2):
  - Load: AutoModel.from_pretrained(..., trust_remote_code=True) + flash_attention_2
  - Infer: model.infer(tokenizer, prompt='', image_file=path,
                       base_size=N, image_size=M, crop_mode=bool)
  - Budget modes control image resolution → visual token count:

    Budget  | base_size | image_size | crop_mode | ~vision tokens
    --------|-----------|------------|-----------|---------------
    tiny    |   512     |    512     |  False    |  ~256
    small   |   640     |    640     |  False    |  ~400
    base    |  1024     |   1024     |  False    |  ~1024
    gundam  |  1024     |    640     |   True    |  adaptive (highest quality)

  DeepSeek-OCR-2 dynamic resolution: (0-6)×768×768 + 1×1024×1024
  → (0-6)×144 + 256 visual tokens.

  Sources:
    https://huggingface.co/deepseek-ai/DeepSeek-OCR
    https://huggingface.co/deepseek-ai/DeepSeek-OCR-2
    https://arxiv.org/abs/2601.20552

REQUIREMENTS:
  torch>=2.6.0, transformers==4.46.3, flash-attn>=2.7.3
  (DeepSeek-OCR-2 specifies these versions — pin them in requirements.txt)

Usage:
  # Gate A — scrambled text, all budgets
  python infer_deepseek.py \\
      --manifest data/manifests/gate.jsonl \\
      --out results/gate_a_deepseek.jsonl \\
      --budget tiny small base gundam

  # Gate B — real text only (reuses same run for efficiency)
  python infer_deepseek.py \\
      --manifest data/manifests/gate.jsonl \\
      --out results/gate_b_deepseek.jsonl \\
      --mode real --budget base
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Canonical model IDs — verify at https://huggingface.co/deepseek-ai before running
DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-OCR-2"  # v2 preferred; fall back to DeepSeek-OCR

# Budget modes → model.infer() kwargs.
# base_size controls the base resolution grid; image_size controls the per-tile
# resolution; crop_mode=True enables adaptive multi-crop (Gundam = highest quality).
BUDGET_CONFIG: dict[str, dict] = {
    "tiny":   {"base_size": 512,  "image_size": 512,  "crop_mode": False},
    "small":  {"base_size": 640,  "image_size": 640,  "crop_mode": False},
    "base":   {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "gundam": {"base_size": 1024, "image_size": 640,  "crop_mode": True},
}

# Approximate visual token counts per budget (used as x-axis for Pillar 3).
# Measure empirically if possible (see _estimate_vision_tokens below).
BUDGET_VTOKENS_APPROX: dict[str, int] = {
    "tiny":   256,
    "small":  400,
    "base":   1024,
    "gundam": 1300,  # adaptive; mid-estimate
}


def _estimate_vision_tokens(base_size: int, image_size: int, crop_mode: bool) -> int:
    """
    Rough estimate of visual token count from resolution parameters.
    Replace with actual measured count if you can hook into model internals.
    For DeepSeek-OCR-2: tiles of 768×768 give 144 tokens each; 1024×1024 tile = 256.
    """
    if crop_mode:
        # Adaptive: up to 6 tiles × 144 + 1 × 256
        return 6 * 144 + 256
    patch = 32  # approximate patch size
    n = (base_size // patch) ** 2
    return n


def load_model(model_id: str, device: str = "cuda"):
    """Load DeepSeek-OCR model via AutoModel (trust_remote_code required)."""
    import torch
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise ImportError("transformers>=4.40 required: pip install transformers") from e

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Use flash_attention_2 if available, fall back to eager (still fast on A100)
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"
        print("  flash_attn not installed — using eager attention (slightly slower)")

    print(f"Loading model: {model_id}  (bfloat16, {attn_impl})")
    model = AutoModel.from_pretrained(
        model_id,
        _attn_implementation=attn_impl,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().to(device)
    return model, tokenizer


def infer_one(
    model,
    tokenizer,
    image_path: str,
    budget_name: str,
) -> tuple[str, int]:
    """
    Run OCR on one image at a given budget.
    Returns (prediction_text, estimated_vision_token_count).
    """
    cfg = BUDGET_CONFIG[budget_name]
    result = model.infer(
        tokenizer,
        prompt="",           # empty prompt → pure OCR mode
        image_file=str(image_path),
        base_size=cfg["base_size"],
        image_size=cfg["image_size"],
        crop_mode=cfg["crop_mode"],
        test_compress=False,
        save_results=False,
    )
    # model.infer returns a string (the transcription)
    prediction = result.strip() if isinstance(result, str) else str(result).strip()
    vtokens = _estimate_vision_tokens(cfg["base_size"], cfg["image_size"], cfg["crop_mode"])
    return prediction, vtokens


def run_inference(
    manifest_path: Path,
    out_path: Path,
    budgets: list[str],
    mode_filter: str | None,
    script_filter: list[str] | None,
    device: str = "cuda",
    model_id: str = DEFAULT_MODEL_ID,
) -> None:
    model, tokenizer = load_model(model_id, device)

    records: list[dict] = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if mode_filter and rec.get("mode") != mode_filter:
                continue
            if script_filter and rec.get("script") not in script_filter:
                continue
            records.append(rec)

    print(f"Inference: {len(records)} records × {len(budgets)} budgets "
          f"= {len(records) * len(budgets)} calls")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(records) * len(budgets)
    done = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for rec in records:
            img_path = rec.get("image_path", "")
            if not img_path or not Path(img_path).exists():
                print(f"  SKIP [{rec['id']}]: image missing at '{img_path}'")
                continue
            for budget in budgets:
                t0 = time.time()
                try:
                    pred, vtokens = infer_one(model, tokenizer, img_path, budget)
                    error = None
                except Exception as exc:
                    pred = ""
                    vtokens = BUDGET_VTOKENS_APPROX.get(budget, -1)
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
    ap.add_argument("--manifest",  required=True)
    ap.add_argument("--out",       required=True)
    ap.add_argument("--budget",    nargs="+", default=["tiny", "small", "base", "gundam"],
                    choices=list(BUDGET_CONFIG.keys()))
    ap.add_argument("--mode",      default=None, choices=["real", "scrambled"])
    ap.add_argument("--script",    nargs="+", default=None)
    ap.add_argument("--device",    default="cuda")
    ap.add_argument("--model_id",  default=DEFAULT_MODEL_ID,
                    help="HF model ID; use deepseek-ai/DeepSeek-OCR for v1")
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
