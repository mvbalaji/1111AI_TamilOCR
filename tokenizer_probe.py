"""
tokenizer_probe.py — measure decoder tokenizer fragmentation (CPU).

Established empirical baseline (Section 4 of project spec):
  Latin  ≈ 0.24 tokens/grapheme  (both tokenizers)
  Tamil  — DeepSeek-OCR 0.96  vs  Qwen3-VL-2B 1.82
  Devanagari — DeepSeek 1.19  vs  Qwen 2.17

These numbers are the tokenizer confound that must be *disentangled* from the
vision confound in Pillar 3.  The vision-only scrambled probe removes the
language-prior confound; the tokens/grapheme x-axis accounts for the tokenizer
confound.

Usage:
  python tokenizer_probe.py --model deepseek-ocr
  python tokenizer_probe.py --model qwen3-vl

Tokenizer IDs (verify on HF before the A100 session — model IDs drift):
  deepseek-ocr : "deepseek-ai/deepseek-ocr"  (update if slug changed)
  qwen3-vl     : "Qwen/Qwen3-VL-2B-Instruct"  (check HF for latest)
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from textkit import normalize, segment

# Model → HF tokenizer id map.  Verify these before running on A100.
TOKENIZER_IDS = {
    "deepseek-ocr": "deepseek-ai/DeepSeek-OCR-2",
    "qwen3-vl":     "Qwen/Qwen3-VL-2B-Instruct",
}

SAMPLE_TEXTS = {
    "tamil": [
        "தமிழ் மொழி உலகின் பழமையான மொழிகளில் ஒன்றாகும்",
        "வணக்கம் நண்பர்களே இன்று மகிழ்ச்சியான நாள்",
        "பள்ளி மாணவர்கள் தினமும் கல்வி கற்கின்றனர்",
        "இந்திய அரசியல் சாசனம் அனைவருக்கும் சம உரிமை அளிக்கிறது",
        "சென்னை தமிழ்நாட்டின் தலைநகரமாகும்",
    ],
    "devanagari": [
        "हिंदी भारत की राजभाषा है",
        "विज्ञान और प्रौद्योगिकी ने जीवन को सरल बनाया है",
        "भारतीय संविधान सभी नागरिकों को समान अधिकार देता है",
        "दिल्ली भारत की राजधानी है",
        "पुस्तकालय में हजारों पुस्तकें उपलब्ध हैं",
    ],
    "latin": [
        "The quick brown fox jumps over the lazy dog",
        "Science and technology have transformed modern life",
        "Libraries contain thousands of books and resources",
        "Children go to school every day to learn new things",
        "Rainwater harvesting is an important method",
    ],
}


def _load_tokenizer(tok_id: str):
    """
    Load a HF tokenizer robustly.

    Strategy (in order):
      1. AutoTokenizer without trust_remote_code — works for standard tokenizers
         even when the *model* class needs trust_remote_code.
      2. AutoTokenizer with trust_remote_code=True — for custom tokenizer classes.
      3. PreTrainedTokenizerFast directly — last resort when model code has broken
         imports (e.g. LlamaFlashAttention2 removed in newer transformers).
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    # Attempt 1: no trust_remote_code (avoids importing broken model code)
    try:
        return AutoTokenizer.from_pretrained(tok_id, trust_remote_code=False)
    except Exception:
        pass

    # Attempt 2: with trust_remote_code (needed if tokenizer class itself is custom)
    try:
        return AutoTokenizer.from_pretrained(tok_id, trust_remote_code=True)
    except ImportError as e:
        if "LlamaFlashAttention2" not in str(e) and "cannot import" not in str(e):
            raise
        # The model code (not the tokenizer) has a broken import from newer transformers.
        # Fall through to direct load.
        print(f"  NOTE: model class import failed ({e}); loading tokenizer files directly.")

    # Attempt 3: load tokenizer.json / tokenizer_config.json directly from cache/hub
    # snapshot_download pulls just the tokenizer files (no model weights).
    try:
        from huggingface_hub import snapshot_download
        local = snapshot_download(
            tok_id,
            ignore_patterns=["*.bin", "*.safetensors", "*.pt", "*.ot", "flax_*"],
        )
        return PreTrainedTokenizerFast.from_pretrained(local)
    except Exception as e2:
        raise RuntimeError(
            f"All tokenizer load strategies failed for {tok_id}.\n"
            f"Last error: {e2}\n"
            "Try: pip install huggingface_hub"
        ) from e2


def measure_fragmentation(model_key: str, verbose: bool = False) -> dict[str, float]:
    """
    Returns {script: mean_tokens_per_grapheme} for the three scripts.
    """
    try:
        from transformers import AutoTokenizer  # noqa: F401
    except ImportError as e:
        raise ImportError("transformers required: pip install transformers") from e

    tok_id = TOKENIZER_IDS[model_key]
    print(f"Loading tokenizer: {tok_id}")
    tokenizer = _load_tokenizer(tok_id)

    results: dict[str, float] = {}
    for script, texts in SAMPLE_TEXTS.items():
        ratios = []
        for text in texts:
            text_nfc = normalize(text)
            n_graphemes = len(segment(text_nfc))
            token_ids = tokenizer.encode(text_nfc, add_special_tokens=False)
            n_tokens = len(token_ids)
            ratio = n_tokens / n_graphemes if n_graphemes > 0 else 0.0
            ratios.append(ratio)
            if verbose:
                print(f"  {script}: '{text_nfc[:30]}...'  "
                      f"graphemes={n_graphemes} tokens={n_tokens} ratio={ratio:.3f}")
        mean_ratio = statistics.mean(ratios)
        results[script] = mean_ratio
        print(f"  {script:>12}: {mean_ratio:.3f} tokens/grapheme  "
              f"(n={len(texts)} samples, stdev={statistics.stdev(ratios):.3f})")

    return results


def compare_tokenizers() -> dict[str, dict[str, float]]:
    """Run both tokenizers and return nested dict for analysis."""
    all_results: dict[str, dict[str, float]] = {}
    for model_key in TOKENIZER_IDS:
        print(f"\n=== {model_key} ===")
        all_results[model_key] = measure_fragmentation(model_key)

    print("\n--- Comparison table (tokens/grapheme) ---")
    scripts = ["tamil", "devanagari", "latin"]
    print(f"{'script':>12} | " + " | ".join(f"{k:>14}" for k in TOKENIZER_IDS))
    for sc in scripts:
        row = [f"{all_results[k].get(sc, float('nan')):.3f}" for k in TOKENIZER_IDS]
        print(f"{sc:>12} | " + " | ".join(f"{v:>14}" for v in row))

    return all_results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default=None,
                    choices=list(TOKENIZER_IDS.keys()) + ["all"],
                    help="tokenizer to probe; 'all' runs both")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.model == "all" or args.model is None:
        compare_tokenizers()
    else:
        measure_fragmentation(args.model, verbose=args.verbose)


if __name__ == "__main__":
    main()
