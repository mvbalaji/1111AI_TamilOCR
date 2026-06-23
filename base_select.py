"""
base_select.py — Gate B: base model selection for Path A+.

Decision rule (Section 6, Gate B):
  dCER = CER_Qwen − CER_DeepSeek  (on real Tamil text, ≥150 lines, ≥2 seeds)

  dCER ≥ −0.03 → DeepSeek-OCR  (tie → fragmentation + Pillar-3 coherence)
  dCER ≤ −0.05 → Qwen3-VL      (clearly ≥5pts better; fix fragmentation in Path A+)
  between      → BORDERLINE     (collect more data; default to DeepSeek for coherence)

Inputs:
  --deepseek_results  results/gate_b_deepseek.jsonl   (real mode, all scripts)
  --qwen_results      results/gate_b_qwen.jsonl        (real mode, all scripts)
  --fragmentation     results/tokenizer_fragmentation.json  (from tokenizer_probe.py)

Usage:
  python base_select.py \
      --deepseek_results results/gate_b_deepseek.jsonl \
      --qwen_results     results/gate_b_qwen.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate import grapheme_cer

MIN_SAMPLES = 150  # minimum per script for a trustworthy verdict


def load_cer_by_script(path: Path, mode_filter: str = "real") -> dict[str, list[float]]:
    """Load predictions from an infer_*.py JSONL and compute per-script CER lists."""
    by_script: dict[str, list[float]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if mode_filter and rec.get("mode") != mode_filter:
                continue
            script = rec.get("script", "unknown")
            pred   = rec.get("prediction", "")
            gt     = rec.get("ground_truth", "")
            cer    = grapheme_cer(pred, gt)
            by_script.setdefault(script, []).append(cer)
    return by_script


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def decide(
    dcer_tamil: float,
    n_deepseek: int,
    n_qwen: int,
    frag: dict | None,
) -> dict:
    """
    dCER = CER_Qwen − CER_DeepSeek.
    Positive → DeepSeek better; Negative → Qwen better.
    """
    n_ok = min(n_deepseek, n_qwen) >= MIN_SAMPLES

    if not n_ok:
        verdict = "INSUFFICIENT_DATA"
        reason  = (f"Need ≥{MIN_SAMPLES} samples per model; "
                   f"got DeepSeek={n_deepseek}, Qwen={n_qwen}. Re-run with more data.")
        choice  = "DEFER"
    elif dcer_tamil >= -0.03:
        verdict = "DEEPSEEK_OCR"
        choice  = "DeepSeek-OCR"
        reason  = (f"dCER={dcer_tamil:.4f} ≥ −0.03 → tie/DeepSeek ahead. "
                   f"DeepSeek also has 2× better Tamil tokenizer fragmentation "
                   f"(0.96 vs 1.82 tokens/grapheme) — Pillar-3 coherent choice.")
    elif dcer_tamil <= -0.05:
        verdict = "QWEN3_VL"
        choice  = "Qwen3-VL-2B"
        reason  = (f"dCER={dcer_tamil:.4f} ≤ −0.05 → Qwen clearly ≥5pts better on Tamil. "
                   f"Fragmentation gap (1.82 vs 0.96) must be fixed in Path A+ via "
                   f"tokenizer extension + embedding retraining.")
    else:
        verdict = "BORDERLINE"
        choice  = "DeepSeek-OCR (default)"
        reason  = (f"dCER={dcer_tamil:.4f} between −0.03 and −0.05 → collect ≥2 more seeds. "
                   f"Defaulting to DeepSeek for Pillar-3 coherence.")

    result = {
        "verdict":        verdict,
        "selected_base":  choice,
        "reason":         reason,
        "dCER_tamil":     round(dcer_tamil, 4),
        "n_deepseek":     n_deepseek,
        "n_qwen":         n_qwen,
        "sufficient_n":   n_ok,
    }
    if frag:
        result["tokenizer_fragmentation"] = frag
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deepseek_results", required=True)
    ap.add_argument("--qwen_results",     required=True)
    ap.add_argument("--fragmentation",    default=None,
                    help="JSON from tokenizer_probe.py (optional)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds_by_script  = load_cer_by_script(Path(args.deepseek_results))
    qw_by_script  = load_cer_by_script(Path(args.qwen_results))

    print("\nPer-script grapheme-CER (real text):")
    print(f"  {'script':>12} | {'DeepSeek':>10} | {'Qwen3-VL':>10} | {'dCER':>10} | {'n_ds':>6} | {'n_qw':>6}")
    print("  " + "-" * 68)

    for script in ["tamil", "devanagari", "latin"]:
        ds_vals = ds_by_script.get(script, [])
        qw_vals = qw_by_script.get(script, [])
        ds_cer  = mean(ds_vals)
        qw_cer  = mean(qw_vals)
        dcer    = qw_cer - ds_cer
        print(f"  {script:>12} | {ds_cer:>10.4f} | {qw_cer:>10.4f} | {dcer:>10.4f} | {len(ds_vals):>6} | {len(qw_vals):>6}")

    # Tamil is the decision axis
    ds_tamil = ds_by_script.get("tamil", [])
    qw_tamil = qw_by_script.get("tamil", [])
    dcer_tamil = mean(qw_tamil) - mean(ds_tamil)

    frag = None
    if args.fragmentation:
        with open(args.fragmentation) as f:
            frag = json.load(f)

    result = decide(dcer_tamil, len(ds_tamil), len(qw_tamil), frag)

    print(f"\n{'='*60}")
    print(f"GATE B VERDICT: {result['verdict']}")
    print(f"Selected base:  {result['selected_base']}")
    print(f"Reason: {result['reason']}")
    print(f"{'='*60}\n")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Verdict JSON: {args.out}")


if __name__ == "__main__":
    main()
