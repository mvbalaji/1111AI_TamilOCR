"""
gate_b.py — Gate B: base model selection (DeepSeek-OCR-2 vs Qwen3-VL).

Decision rule:
  dCER = CER_Qwen − CER_DeepSeek  (Tamil, real text, base budget)

  dCER ≥ −0.03 → DeepSeek-OCR-2   (tie → fragmentation advantage wins)
  dCER ≤ −0.05 → Qwen3-VL-2B      (clearly ≥5pts better; fix fragmentation in Path A+)
  between      → BORDERLINE        (default to DeepSeek for Pillar-3 coherence)

Inputs: both models must have been run on the same manifest in real mode,
any budget (base recommended).

Usage:
  python gate_b.py \\
      --deepseek results/gate_a_deepseek.jsonl \\
      --qwen     results/gate_a_qwen.jsonl \\
      --budget   base \\
      --out      results/gate_b_verdict.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate import grapheme_cer

MIN_SAMPLES = 150


def load_cer_by_script(path: Path, mode_filter: str = "real",
                        budget_filter: str | None = None) -> dict[str, list[float]]:
    by_script: dict[str, list[float]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if mode_filter and rec.get("mode") != mode_filter:
                continue
            if budget_filter and rec.get("budget") != budget_filter:
                continue
            script = rec.get("script", "unknown")
            pred   = rec.get("prediction", "")
            gt     = rec.get("ground_truth", "")
            by_script.setdefault(script, []).append(grapheme_cer(pred, gt))
    return by_script


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def decide(dcer_tamil: float, n_ds: int, n_qw: int) -> dict:
    n_ok = min(n_ds, n_qw) >= MIN_SAMPLES

    if not n_ok:
        return {
            "verdict":       "INSUFFICIENT_DATA",
            "selected_base": "DEFER",
            "reason":        (f"Need ≥{MIN_SAMPLES} real samples per model; "
                              f"got DeepSeek={n_ds}, Qwen={n_qw}. Re-run with more data."),
            "dCER_tamil":    round(dcer_tamil, 4),
            "n_deepseek":    n_ds,
            "n_qwen":        n_qw,
            "sufficient_n":  False,
        }

    if dcer_tamil >= -0.03:
        verdict = "DEEPSEEK_OCR"
        choice  = "DeepSeek-OCR-2"
        reason  = (f"dCER={dcer_tamil:.4f} ≥ −0.03 → tie/DeepSeek ahead. "
                   f"DeepSeek also has 2× better Tamil tokenizer efficiency "
                   f"(0.841 vs 1.685 tok/grapheme) — Pillar-3 coherent choice.")
    elif dcer_tamil <= -0.05:
        verdict = "QWEN3_VL"
        choice  = "Qwen3-VL-2B"
        reason  = (f"dCER={dcer_tamil:.4f} ≤ −0.05 → Qwen clearly ≥5pts better on Tamil. "
                   f"Tokenizer gap (1.685 vs 0.841 tok/g) must be addressed in Path A+ "
                   f"via tokenizer extension + embedding retraining.")
    else:
        verdict = "BORDERLINE"
        choice  = "DeepSeek-OCR-2 (default)"
        reason  = (f"dCER={dcer_tamil:.4f} between −0.03 and −0.05 → borderline. "
                   f"Collect ≥2 more seeds. Defaulting to DeepSeek for Pillar-3 coherence.")

    return {
        "verdict":       verdict,
        "selected_base": choice,
        "reason":        reason,
        "dCER_tamil":    round(dcer_tamil, 4),
        "n_deepseek":    n_ds,
        "n_qwen":        n_qw,
        "sufficient_n":  True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deepseek", required=True,
                    help="JSONL from infer_deepseek.py")
    ap.add_argument("--qwen",     required=True,
                    help="JSONL from infer_qwen.py")
    ap.add_argument("--budget",   default="base",
                    choices=["tiny", "small", "base", "gundam"],
                    help="which budget tier to compare (default: base)")
    ap.add_argument("--out",      default=None)
    args = ap.parse_args()

    ds = load_cer_by_script(Path(args.deepseek), budget_filter=args.budget)
    qw = load_cer_by_script(Path(args.qwen),     budget_filter=args.budget)

    print(f"\nGate B comparison (budget={args.budget}, mode=real, grapheme-CER):")
    print(f"  {'script':>12} | {'DeepSeek':>10} | {'Qwen3-VL':>10} | {'dCER':>10} | n_ds | n_qw")
    print("  " + "-" * 68)

    for script in ["tamil", "devanagari", "latin"]:
        ds_vals = ds.get(script, [])
        qw_vals = qw.get(script, [])
        ds_cer  = _mean(ds_vals)
        qw_cer  = _mean(qw_vals)
        dcer    = qw_cer - ds_cer
        print(f"  {script:>12} | {ds_cer:>10.4f} | {qw_cer:>10.4f} | {dcer:>10.4f} "
              f"| {len(ds_vals):>4} | {len(qw_vals):>4}")

    ds_tamil   = ds.get("tamil", [])
    qw_tamil   = qw.get("tamil", [])
    dcer_tamil = _mean(qw_tamil) - _mean(ds_tamil)

    result = decide(dcer_tamil, len(ds_tamil), len(qw_tamil))

    print(f"\n{'='*60}")
    print(f"GATE B VERDICT: {result['verdict']}")
    print(f"Selected base:  {result['selected_base']}")
    print(f"Reason: {result['reason']}")
    print(f"{'='*60}\n")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Verdict JSON → {args.out}")


if __name__ == "__main__":
    main()
