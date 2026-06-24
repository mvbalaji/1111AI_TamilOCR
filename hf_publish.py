"""
hf_publish.py — push benchmark dataset and/or trained model to HuggingFace.

Two subcommands:

  python hf_publish.py dataset             # push benchmark data + eval harness
  python hf_publish.py model --checkpoint_dir path/to/checkpoint

HF destinations (HF_USERNAME can be overridden via --username):
  Dataset : mvbalaji/tamil-ocr-benchmark
  Model   : mvbalaji/tamil-ocr-vlm

Prerequisites:
  pip install huggingface_hub
  huggingface-cli login          (write-access token required)

LICENSE COMPLIANCE GUARD
  DeepSeek-OCR-2 : "DeepSeek Model License" — derivative model weights must
                   carry the same license and may not be used to train models
                   that compete with DeepSeek products.  REVIEW before pushing.
  Qwen3-VL       : Apache 2.0 — permissive, derivative works OK.
  The script will refuse to push a model derived from DeepSeek-OCR-2 unless
  you explicitly pass --ack_deepseek_license.

Usage examples:

  # Push dataset after Gate experiments (works with partial data too)
  python hf_publish.py dataset \\
      --manifest data/manifests/gate.jsonl \\
      --results_dir results/ \\
      --pillar3_verdict results/pillar3_verdict.json

  # Push trained model checkpoint (after P4)
  python hf_publish.py model \\
      --checkpoint_dir checkpoints/tamil-ocr-vlm-v1 \\
      --base_model deepseek-ai/DeepSeek-OCR-2 \\
      --ack_deepseek_license          # required if base is DeepSeek

  # Dry run (prints what would be pushed, nothing uploaded)
  python hf_publish.py dataset --dry_run
  python hf_publish.py model --checkpoint_dir ... --dry_run
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_USERNAME    = "mvbalaji"
DATASET_REPO   = f"{HF_USERNAME}/tamil-ocr-benchmark"
MODEL_REPO     = f"{HF_USERNAME}/tamil-ocr-vlm"

# Files from the harness that are part of the eval artefact (always included)
HARNESS_FILES = [
    "textkit.py",
    "evaluate.py",
    "benchmark_spec.py",
    "pillar2_demo.py",
    "README.md",
    "requirements.txt",
]

DEEPSEEK_LICENSE_WARNING = """
WARNING — DeepSeek Model License detected.
The DeepSeek Model License restricts derivative model weights:
  - Derivative weights must carry the same license.
  - May not be used to train models that compete with DeepSeek products.
  - Commercial use requires separate agreement above 100M MAU.

Review the full license at:
  https://huggingface.co/deepseek-ai/DeepSeek-OCR-2/blob/main/LICENSE

If you have reviewed and accept these terms, re-run with --ack_deepseek_license.
"""

# ---------------------------------------------------------------------------
# Dataset card
# ---------------------------------------------------------------------------

def make_dataset_card(
    pillar3_verdict: dict | None,
    base_select_verdict: dict | None,
    n_records: int,
) -> str:
    pillar3_str = (pillar3_verdict.get("verdict")
                   or pillar3_verdict.get("gate_a_verdict")
                   or "pending") if pillar3_verdict else "pending"
    base_str    = base_select_verdict["selected_base"] if base_select_verdict else "pending"

    return f"""\
---
language:
  - ta
  - hi
  - en
license: cc-by-4.0
task_categories:
  - image-to-text
task_ids:
  - optical-character-recognition
tags:
  - tamil
  - ocr
  - benchmark
  - indic-nlp
  - grapheme-cer
  - compression-density
pretty_name: Tamil OCR Benchmark v1
size_categories:
  - 1K<n<10K
---

# Tamil OCR Benchmark v1

The **first independent, open benchmark** of 2026-generation OCR-VLMs on Tamil,
with structure-coverage-controlled evaluation and a grapheme-aware metric protocol.

## Highlights

- **247-grapheme coverage matrix** — full uyirmey grid + Grantha + split matras +
  Tamil numerals
- **Grapheme-cluster CER** (primary metric) — edit distance over `\\X` clusters
  after NFC normalisation, shown to be unbiased vs. codepoint-CER for Indic scripts
- **Compression × script-density study** (Pillar 3) — Pillar 3 verdict: **{pillar3_str}**
- **Document tiers v1**: printed multi-column, tables/forms, Tanglish
- **Vision-only scrambled probe** — removes decoder language prior to isolate
  visual confound in compression experiments

## Gate results

| Gate | Verdict |
|---|---|
| Pillar 3 (compression × density) | {pillar3_str} |
| Base model selection | {base_str} |

## Dataset structure

```
data/
  images/<split>/<script>/<real|scrambled>/<idx>.png
  manifests/<split>.jsonl
    fields: id, split, script, mode, text, ground_truth,
            image_path, grapheme_count
results/
  gate_a_deepseek.jsonl    DeepSeek-OCR-2 predictions (all budgets)
  gate_b_qwen.jsonl        Qwen3-VL predictions (real mode)
  pillar3_verdict.json     Gate A decision
  base_select_verdict.json Gate B decision
eval/
  evaluate.py              grapheme_cer + codepoint_cer
  benchmark_spec.py        coverage matrix + metric definitions
  textkit.py               grapheme segmentation utilities
```

## Metrics

| Metric | Description |
|---|---|
| `grapheme_cer` | Edit distance over `\\X` grapheme clusters / cluster count (primary) |
| `codepoint_cer` | Edit distance over NFC codepoints / codepoint count (Pillar 2 demo only) |
| `word_acc` | Fraction of reference words matched at same position |
| `teds` | Tree Edit Distance Score for table/form tier |

## Records

{n_records} image–text pairs across {3} scripts × 2 rendering modes (real / scrambled).

## Citation

```bibtex
@misc{{tamil-ocr-benchmark-2026,
  title  = {{Tamil OCR Benchmark v1: Compression, Density, and Script}},
  author = {{Venkateswaran, Balaji}},
  year   = {{2026}},
  url    = {{https://huggingface.co/datasets/{DATASET_REPO}}}
}}
```

## License

Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
Fonts: [OFL 1.1](https://scripts.sil.org/OFL) (Noto Sans family, Google)
Code: Apache 2.0
"""


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------

def make_model_card(
    base_model: str,
    scores: dict | None,
    pillar3_verdict: dict | None,
) -> str:
    pillar3_str = pillar3_verdict["verdict"] if pillar3_verdict else "pending"
    scores_md = ""
    if scores:
        scores_md = "## Benchmark scores\n\n| Split | grapheme-CER | word-acc |\n|---|---|---|\n"
        for split, vals in scores.items():
            scores_md += (f"| {split} | {vals.get('grapheme_cer', 'N/A'):.4f} | "
                          f"{vals.get('word_acc', 'N/A'):.4f} |\n")

    license_id = "apache-2.0" if "Qwen" in base_model else "deepseek"

    return f"""\
---
language:
  - ta
license: {license_id}
base_model: {base_model}
tags:
  - tamil
  - ocr
  - vision-language-model
  - indic-nlp
pipeline_tag: image-to-text
---

# Tamil OCR VLM

A Tamil-dedicated OCR vision-language model fine-tuned from `{base_model}`.

Targets **parameter efficiency**: a ≤2B model matching or beating 3B multilingual
baselines on targeted Tamil OCR distributions (printed multi-column, tables/forms,
Tanglish).

## Training summary

- **Base model**: `{base_model}`
- **Fine-tune stages**:
  1. Vision encoder unfreezing (Tamil glyphs novel to encoder)
  2. Tokenizer extension with Tamil akshara tokens + embedding retraining
  3. Decoder Tamil prior (continued pretraining / distillation)
  4. Document SFT on Tamil OCR Benchmark v1 synthetic data
- **Pillar 3 motivation**: {pillar3_str} — compression degrades Tamil
  disproportionately; this model addresses the visual encoding gap.

{scores_md}

## Limitations

- Trained on synthetic + limited real Tamil text; may not generalise to
  handwriting, palm-leaf manuscripts, or highly degraded scans.
- Tanglish (code-switched) coverage is limited to printed documents.
- Evaluated on Tamil only — performance on other Indic scripts is unknown.

## Citation

```bibtex
@misc{{tamil-ocr-vlm-2026,
  title  = {{Tamil OCR VLM: A Tamil-Dedicated OCR Vision-Language Model}},
  author = {{Venkateswaran, Balaji}},
  year   = {{2026}},
  url    = {{https://huggingface.co/models/{MODEL_REPO}}}
}}
```

## License

Derived from `{base_model}` — see base model license for derivative use terms.
"""


# ---------------------------------------------------------------------------
# Push helpers
# ---------------------------------------------------------------------------

def _hub():
    try:
        from huggingface_hub import HfApi
        return HfApi()
    except ImportError as e:
        raise ImportError(
            "huggingface_hub required: pip install huggingface_hub"
        ) from e


def push_dataset(
    manifest_path: Path | None,
    results_dir: Path | None,
    pillar3_verdict_path: Path | None,
    base_select_verdict_path: Path | None,
    repo_id: str,
    dry_run: bool,
) -> None:
    api = _hub()

    # Load verdicts if present
    pillar3  = json.loads(pillar3_verdict_path.read_text())  if pillar3_verdict_path  and pillar3_verdict_path.exists()  else None
    base_sel = json.loads(base_select_verdict_path.read_text()) if base_select_verdict_path and base_select_verdict_path.exists() else None

    # Count records
    n_records = 0
    if manifest_path and manifest_path.exists():
        n_records = sum(1 for _ in open(manifest_path, encoding="utf-8"))

    card = make_dataset_card(pillar3, base_sel, n_records)

    if dry_run:
        print(f"\n[DRY RUN] Would create/update dataset repo: {repo_id}")
        print(f"  manifest records : {n_records}")
        print(f"  Pillar 3 verdict : {pillar3['verdict'] if pillar3 else 'none'}")
        print(f"  Base model       : {base_sel['selected_base'] if base_sel else 'none'}")
        print("\n--- Dataset card preview (first 20 lines) ---")
        for line in card.splitlines()[:20]:
            print(" ", line)
        return

    # Create repo if it doesn't exist
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)
    print(f"Repo ready: https://huggingface.co/datasets/{repo_id}")

    # Upload dataset card
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(card)
        card_path = f.name
    api.upload_file(
        path_or_fileobj=card_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update dataset card",
    )
    print("  Uploaded: README.md (dataset card)")

    # Upload harness eval scripts
    for fname in HARNESS_FILES:
        p = Path(fname)
        if p.exists():
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=f"eval/{fname}",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Upload eval harness: {fname}",
            )
            print(f"  Uploaded: eval/{fname}")

    # Upload manifest
    if manifest_path and manifest_path.exists():
        api.upload_file(
            path_or_fileobj=str(manifest_path),
            path_in_repo=f"data/manifests/{manifest_path.name}",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload manifest",
        )
        print(f"  Uploaded: data/manifests/{manifest_path.name}")

    # Upload results JSONs
    if results_dir and results_dir.exists():
        for jf in sorted(results_dir.glob("*.json*")):
            api.upload_file(
                path_or_fileobj=str(jf),
                path_in_repo=f"results/{jf.name}",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Upload results: {jf.name}",
            )
            print(f"  Uploaded: results/{jf.name}")

    # Upload images folder if it exists (may be large — upload as folder)
    images_dir = Path("data/images")
    if images_dir.exists():
        print(f"  Uploading images/ (this may take a while) ...")
        api.upload_folder(
            folder_path=str(images_dir),
            path_in_repo="data/images",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload benchmark images",
        )
        print(f"  Uploaded: data/images/")

    print(f"\nDataset live at: https://huggingface.co/datasets/{repo_id}")


def push_model(
    checkpoint_dir: Path,
    base_model: str,
    repo_id: str,
    scores_path: Path | None,
    pillar3_verdict_path: Path | None,
    ack_deepseek_license: bool,
    dry_run: bool,
) -> None:
    # License guard
    if "deepseek" in base_model.lower() and not ack_deepseek_license:
        print(DEEPSEEK_LICENSE_WARNING)
        sys.exit(1)

    api = _hub()

    pillar3 = json.loads(pillar3_verdict_path.read_text()) if pillar3_verdict_path and pillar3_verdict_path.exists() else None
    scores  = json.loads(scores_path.read_text())          if scores_path and scores_path.exists()          else None

    card = make_model_card(base_model, scores, pillar3)

    if dry_run:
        print(f"\n[DRY RUN] Would create/update model repo: {repo_id}")
        print(f"  checkpoint_dir : {checkpoint_dir}")
        print(f"  base_model     : {base_model}")
        print(f"  Pillar 3       : {pillar3['verdict'] if pillar3 else 'none'}")
        print("\n--- Model card preview (first 20 lines) ---")
        for line in card.splitlines()[:20]:
            print(" ", line)
        return

    if not checkpoint_dir.exists():
        print(f"ERROR: checkpoint_dir not found: {checkpoint_dir}")
        sys.exit(1)

    # Create repo
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
    print(f"Repo ready: https://huggingface.co/{repo_id}")

    # Upload model card
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(card)
        card_path = f.name
    api.upload_file(
        path_or_fileobj=card_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update model card",
    )
    print("  Uploaded: README.md (model card)")

    # Upload checkpoint folder
    print(f"  Uploading checkpoint from {checkpoint_dir} ...")
    api.upload_folder(
        folder_path=str(checkpoint_dir),
        path_in_repo=".",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload model checkpoint",
        ignore_patterns=["*.pyc", "__pycache__", "*.log"],
    )
    print(f"  Uploaded: checkpoint ({checkpoint_dir})")
    print(f"\nModel live at: https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Push benchmark dataset or trained model to HuggingFace."
    )
    ap.add_argument("--username", default=HF_USERNAME,
                    help=f"HF username (default: {HF_USERNAME})")

    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- dataset subcommand ---
    ds = sub.add_parser("dataset", help="Push benchmark dataset")
    ds.add_argument("--dry_run", action="store_true",
                    help="Print what would be pushed without uploading")
    ds.add_argument("--manifest", default="data/manifests/gate.jsonl",
                    help="JSONL manifest from datagen.py")
    ds.add_argument("--results_dir", default="results/",
                    help="Directory containing gate result JSONs")
    ds.add_argument("--pillar3_verdict", default="results/pillar3_verdict.json")
    ds.add_argument("--base_select_verdict", default="results/base_select_verdict.json")
    ds.add_argument("--repo_id", default=None,
                    help="Override HF dataset repo ID")

    # --- model subcommand ---
    mo = sub.add_parser("model", help="Push trained model checkpoint")
    mo.add_argument("--checkpoint_dir", required=True,
                    help="Local directory containing the trained checkpoint")
    mo.add_argument("--base_model", default="deepseek-ai/DeepSeek-OCR-2",
                    help="HF model ID of the base model used for fine-tuning")
    mo.add_argument("--scores", default=None,
                    help="JSON file with benchmark scores {split: {grapheme_cer, word_acc}}")
    mo.add_argument("--pillar3_verdict", default="results/pillar3_verdict.json")
    mo.add_argument("--ack_deepseek_license", action="store_true",
                    help="Required when base_model is a DeepSeek model")
    mo.add_argument("--dry_run", action="store_true",
                    help="Print what would be pushed without uploading")
    mo.add_argument("--repo_id", default=None,
                    help="Override HF model repo ID")

    args = ap.parse_args()

    if args.cmd == "dataset":
        push_dataset(
            manifest_path         = Path(args.manifest),
            results_dir           = Path(args.results_dir),
            pillar3_verdict_path  = Path(args.pillar3_verdict),
            base_select_verdict_path = Path(args.base_select_verdict),
            repo_id               = args.repo_id or f"{args.username}/tamil-ocr-benchmark",
            dry_run               = args.dry_run,
        )

    elif args.cmd == "model":
        push_model(
            checkpoint_dir        = Path(args.checkpoint_dir),
            base_model            = args.base_model,
            repo_id               = args.repo_id or f"{args.username}/tamil-ocr-vlm",
            scores_path           = Path(args.scores) if args.scores else None,
            pillar3_verdict_path  = Path(args.pillar3_verdict),
            ack_deepseek_license  = args.ack_deepseek_license,
            dry_run               = args.dry_run,
        )


if __name__ == "__main__":
    main()
