"""
finetune.py — Fine-tune Qwen3-VL-2B on Tamil OCR using LoRA + TRL SFTTrainer.

Path A+: Gate B selected Qwen3-VL-2B as base. This script fine-tunes it on
synthetic Tamil OCR data to close the Tamil CER gap (baseline: 0.24 → target: ~0.08).

Strategy:
  - LoRA rank 64 on all attention projections (q/k/v/o) in language decoder
  - Vision encoder unfrozen (Tamil glyphs novel to encoder)
  - Training data: Tamil real-text images from datagen.py (line + multicolumn + table)
  - Loss: cross-entropy on OCR transcription tokens only (image tokens masked)

Requirements (ocr_qwen env):
  pip install trl>=0.9.0 peft>=0.10.0 bitsandbytes>=0.43.0

Data generation (run before training):
  python datagen.py \\
      --out_dir data_train \\
      --corpus_dir corpora \\
      --use_corpora \\
      --n_lines 8000 \\
      --splits train \\
      --multi_font \\
      --aug_level medium

Usage:
  # Full fine-tune run (~4 hours on A100 40GB)
  python finetune.py \\
      --manifest  data_train/manifests/train.jsonl \\
      --out_dir   checkpoints/tamil-ocr-v1 \\
      --script    tamil \\
      --epochs    3

  # Quick smoke test (100 samples, 1 epoch)
  python finetune.py \\
      --manifest  data_train/manifests/train.jsonl \\
      --out_dir   checkpoints/smoke \\
      --script    tamil \\
      --epochs    1 \\
      --max_samples 100

  # Evaluate after training
  python evaluate_ft.py \\
      --manifest   data/manifests/gate.jsonl \\
      --checkpoint checkpoints/tamil-ocr-v1 \\
      --out        results/ft_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_MODEL_ID  = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_OUT    = "checkpoints/tamil-ocr-v1"
OCR_PROMPT     = (
    "Transcribe the text in this image exactly as it appears. "
    "Output only the transcribed text, nothing else."
)

# LoRA targets — all attention projections in the language decoder.
# Vision encoder is unfrozen separately via requires_grad.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TamilOCRDataset(torch.utils.data.Dataset):
    """
    Loads a datagen.py manifest and returns one dict per record.

    script_filter : "tamil" | "all" | comma-separated e.g. "tamil,devanagari"
    replay_scripts: additional scripts to sample for catastrophic-forgetting
                    prevention (e.g. ["devanagari", "latin"])
    replay_ratio  : fraction of primary-script samples to draw from each
                    replay script (0.1 = 10% replay per script)
    """

    def __init__(
        self,
        manifest_path: str,
        script_filter: str = "tamil",
        max_samples: int | None = None,
        replay_scripts: list[str] | None = None,
        replay_ratio: float = 0.1,
    ):
        import random

        # Parse primary script filter
        if script_filter == "all":
            primary_allowed = None
        else:
            primary_allowed = set(script_filter.split(","))

        # Load all records grouped by script
        by_script: dict[str, list[dict]] = {}
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("mode") != "real":
                    continue
                if not Path(rec["image_path"]).exists():
                    continue
                sc = rec.get("script", "")
                by_script.setdefault(sc, []).append(rec)

        # Primary records
        if primary_allowed is None:
            primary = [r for recs in by_script.values() for r in recs]
        else:
            primary = [r for sc in primary_allowed for r in by_script.get(sc, [])]

        if max_samples:
            primary = primary[:max_samples]

        # Replay records: sample replay_ratio × |primary| from each replay script
        replay: list[dict] = []
        if replay_scripts:
            n_replay = max(1, int(len(primary) * replay_ratio))
            rng = random.Random(42)
            for sc in replay_scripts:
                pool = by_script.get(sc, [])
                if not pool:
                    print(f"  WARN: no replay records found for script={sc}")
                    continue
                sampled = rng.choices(pool, k=min(n_replay, len(pool)))
                replay.extend(sampled)
                print(f"  Replay {sc}: {len(sampled)} samples")

        self.records = primary + replay
        rng2 = random.Random(0)
        rng2.shuffle(self.records)

        counts: dict[str, int] = {}
        for r in self.records:
            counts[r.get("script","?")] = counts.get(r.get("script","?"),0) + 1
        print(f"Dataset: {len(self.records)} total (mode=real) — " +
              ", ".join(f"{s}:{n}" for s,n in sorted(counts.items())))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        img = Image.open(rec["image_path"]).convert("RGB")
        return {
            "image":        img,
            "ground_truth": rec["ground_truth"],
            "id":           rec["id"],
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class OCRCollator:
    """
    Formats image+text into Qwen3-VL chat template inputs.
    Labels the transcription tokens for loss; masks the image/prompt tokens.
    Returns CPU tensors — Trainer handles device placement.
    """

    def __init__(self, processor, device: str = "cuda"):
        self.processor = processor
        self.device    = device  # kept for reference, not used in collation

    def __call__(self, batch: list[dict]) -> dict:
        all_inputs = []
        all_labels = []

        for item in batch:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image"]},
                        {"type": "text",  "text": OCR_PROMPT},
                    ],
                },
                {
                    "role": "assistant",
                    "content": item["ground_truth"],
                },
            ]

            # Full sequence (prompt + response)
            full = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )

            # Prompt only (to find where response starts)
            prompt_messages = messages[:1]
            prompt = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            input_ids  = full["input_ids"][0]
            prompt_len = prompt["input_ids"].shape[1]

            # Labels: -100 for prompt tokens, actual ids for response tokens
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            all_inputs.append(full)
            all_labels.append(labels)

        # Pad to max length in batch
        max_len = max(inp["input_ids"].shape[1] for inp in all_inputs)
        pad_id  = self.processor.tokenizer.pad_token_id or 0

        input_ids_batch         = []
        attention_mask_batch    = []
        labels_batch            = []
        pixel_values_batch      = []
        image_grid_thw_batch    = []
        mm_token_type_ids_batch = []

        for inp, lbl in zip(all_inputs, all_labels):
            seq_len = inp["input_ids"].shape[1]
            pad_len = max_len - seq_len

            input_ids_batch.append(
                torch.cat([inp["input_ids"][0],
                           torch.full((pad_len,), pad_id, dtype=torch.long)])
            )
            attention_mask_batch.append(
                torch.cat([inp["attention_mask"][0],
                           torch.zeros(pad_len, dtype=torch.long)])
            )
            labels_batch.append(
                torch.cat([lbl,
                           torch.full((pad_len,), -100, dtype=torch.long)])
            )
            if "pixel_values" in inp:
                pixel_values_batch.append(inp["pixel_values"])
            if "image_grid_thw" in inp:
                image_grid_thw_batch.append(inp["image_grid_thw"])
            if "mm_token_type_ids" in inp:
                mm_token_type_ids_batch.append(
                    torch.cat([inp["mm_token_type_ids"][0],
                               torch.zeros(pad_len, dtype=torch.long)])
                )

        # Return CPU tensors — Trainer moves them to device via its own mechanism
        result = {
            "input_ids":      torch.stack(input_ids_batch),
            "attention_mask": torch.stack(attention_mask_batch),
            "labels":         torch.stack(labels_batch),
        }
        if pixel_values_batch:
            result["pixel_values"] = torch.cat(pixel_values_batch)
        if image_grid_thw_batch:
            result["image_grid_thw"] = torch.cat(image_grid_thw_batch)
        if mm_token_type_ids_batch:
            result["mm_token_type_ids"] = torch.stack(mm_token_type_ids_batch)

        return result


# ---------------------------------------------------------------------------
# LoRA setup
# ---------------------------------------------------------------------------

def apply_lora(model, lora_rank: int = 64, lora_alpha: int = 128):
    from peft import LoraConfig, get_peft_model, TaskType

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def unfreeze_vision_encoder(model) -> None:
    """Unfreeze vision encoder so it learns Tamil-specific glyph features."""
    unfrozen = 0
    for name, param in model.named_parameters():
        if "visual" in name or "vision" in name:
            param.requires_grad = True
            unfrozen += param.numel()
    print(f"Vision encoder unfrozen: {unfrozen/1e6:.1f}M additional parameters")


def freeze_vision_encoder(model) -> None:
    """Keep vision encoder frozen — prevents script-specific visual features
    learned for Tamil from overwriting Devanagari/Latin representations."""
    frozen = 0
    for name, param in model.named_parameters():
        if "visual" in name or "vision" in name:
            param.requires_grad = False
            frozen += param.numel()
    print(f"Vision encoder frozen: {frozen/1e6:.1f}M parameters locked")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    manifest_path: str,
    out_dir: str,
    script_filter: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lora_rank: int,
    max_samples: int | None,
    lr: float,
    device: str,
    replay_scripts: list[str] | None = None,
    replay_ratio: float = 0.1,
    freeze_vision: bool = False,
) -> None:
    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        from transformers import TrainingArguments, Trainer
    except ImportError as e:
        raise ImportError(
            "Install transformers from source:\n"
            "  pip install git+https://github.com/huggingface/transformers"
        ) from e

    try:
        from peft import LoraConfig  # noqa: F401
    except ImportError:
        raise ImportError("pip install peft>=0.10.0")

    replay_label = f" + replay({','.join(replay_scripts)}@{replay_ratio})" if replay_scripts else ""
    vision_label = "frozen" if freeze_vision else "unfrozen"
    print(f"\n{'='*60}")
    print(f"Fine-tuning {BASE_MODEL_ID}")
    print(f"Script: {script_filter}{replay_label} | vision={vision_label}")
    print(f"Epochs: {epochs} | LoRA rank: {lora_rank} | Output: {out_dir}")
    print(f"{'='*60}\n")

    # Load processor
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_ID,
        min_pixels=256 * 28 * 28,
        max_pixels=1024 * 28 * 28,
    )

    # Load model in bfloat16
    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    # Apply LoRA, then handle vision encoder
    model = apply_lora(model, lora_rank=lora_rank, lora_alpha=lora_rank * 2)
    if freeze_vision:
        freeze_vision_encoder(model)
    else:
        unfreeze_vision_encoder(model)
    model.train()

    # Dataset + collator
    dataset  = TamilOCRDataset(
        manifest_path, script_filter, max_samples,
        replay_scripts=replay_scripts,
        replay_ratio=replay_ratio,
    )
    collator = OCRCollator(processor, device=device)

    # Split 90/10 train/eval
    n_eval   = max(1, len(dataset) // 10)
    n_train  = len(dataset) - n_eval
    train_ds, eval_ds = torch.utils.data.random_split(
        dataset, [n_train, n_eval],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {n_train}  Eval: {n_eval}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir                  = str(out_path),
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        gradient_accumulation_steps = grad_accum,
        learning_rate               = lr,
        warmup_ratio                = 0.05,
        lr_scheduler_type           = "cosine",
        bf16                        = True,
        fp16                        = False,
        logging_steps               = 10,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        save_total_limit            = 2,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        report_to                   = "none",
        dataloader_num_workers      = 0,
        dataloader_pin_memory       = False,
        remove_unused_columns       = False,
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = eval_ds,
        data_collator   = collator,
    )

    print("\nStarting training...")
    trainer.train()

    # Save final adapter
    adapter_path = out_path / "adapter_final"
    model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    print(f"\nAdapter saved → {adapter_path}")

    # Save training summary
    summary = {
        "base_model":      BASE_MODEL_ID,
        "script":          script_filter,
        "replay_scripts":  replay_scripts or [],
        "replay_ratio":    replay_ratio,
        "freeze_vision":   freeze_vision,
        "epochs":          epochs,
        "lora_rank":       lora_rank,
        "n_train":         n_train,
        "n_eval":          n_eval,
        "adapter_path":    str(adapter_path),
    }
    with open(out_path / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {out_path / 'train_summary.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",     required=True,
                    help="JSONL manifest from datagen.py (train split)")
    ap.add_argument("--out_dir",      default=DEFAULT_OUT)
    ap.add_argument("--script",          default="tamil",
                    help="primary script(s): tamil|all|comma-separated (default: tamil)")
    ap.add_argument("--replay_scripts",  default="devanagari,latin",
                    help="comma-separated scripts to replay for anti-forgetting "
                         "(default: devanagari,latin; pass '' to disable)")
    ap.add_argument("--replay_ratio",    type=float, default=0.1,
                    help="replay samples per primary sample per replay script "
                         "(default: 0.1 = 10%%)")
    ap.add_argument("--freeze_vision",   action="store_true",
                    help="freeze vision encoder (recommended when using replay to "
                         "prevent cross-script visual feature corruption)")
    ap.add_argument("--epochs",          type=int,   default=3)
    ap.add_argument("--batch_size",   type=int,   default=1,
                    help="per-device batch size (keep 1 for A100 40GB with LoRA)")
    ap.add_argument("--grad_accum",   type=int,   default=16,
                    help="gradient accumulation steps (effective batch = 16)")
    ap.add_argument("--lora_rank",    type=int,   default=64)
    ap.add_argument("--lr",           type=float, default=2e-4)
    ap.add_argument("--max_samples",  type=int,   default=None,
                    help="limit samples for smoke test")
    ap.add_argument("--device",       default="cuda")
    args = ap.parse_args()

    replay = [s.strip() for s in args.replay_scripts.split(",") if s.strip()] \
             if args.replay_scripts else []

    train(
        manifest_path  = args.manifest,
        out_dir        = args.out_dir,
        script_filter  = args.script,
        epochs         = args.epochs,
        batch_size     = args.batch_size,
        grad_accum     = args.grad_accum,
        lora_rank      = args.lora_rank,
        max_samples    = args.max_samples,
        lr             = args.lr,
        device         = args.device,
        replay_scripts = replay,
        replay_ratio   = args.replay_ratio,
        freeze_vision  = args.freeze_vision,
    )


if __name__ == "__main__":
    main()
