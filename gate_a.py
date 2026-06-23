"""
gate_a.py — Gate A: Pillar-3 GO / NO-GO / REFINE decision.

Reads DeepSeek (and optionally Qwen) inference results across all budget tiers
and applies the Gate A criteria:

  GO     — Tamil−Latin scrambled CER gap ≥ 0.10
            AND ≥ 50% of the real-text gap
            AND gap grows as budget tightens (tiny > small > base)

  NO-GO  — gap < 0.05 OR < 25% of real-text gap

  REFINE — everything else (borderline)

Usage:
  # DeepSeek only (primary Gate A decision)
  python gate_a.py \\
      --deepseek results/gate_a_deepseek.jsonl \\
      --out      results/gate_a_verdict.json

  # Both models (full comparison for paper)
  python gate_a.py \\
      --deepseek results/gate_a_deepseek.jsonl \\
      --qwen     results/gate_a_qwen.jsonl \\
      --out      results/gate_a_verdict.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evaluate import grapheme_cer

BUDGET_ORDER = ["tiny", "small", "base", "gundam"]


def load_results(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def compute_cer_table(rows: list[dict], mode: str) -> dict[str, dict[str, float]]:
    """Returns {budget: {script: mean_grapheme_cer}} for one mode."""
    bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("mode") != mode:
            continue
        budget = row.get("budget", "base")
        script = row.get("script", "unknown")
        pred   = row.get("prediction", "")
        gt     = row.get("ground_truth", "")
        bucket[budget][script].append(grapheme_cer(pred, gt))

    return {
        budget: {sc: sum(v) / len(v) for sc, v in scripts.items() if v}
        for budget, scripts in bucket.items()
    }


def monotone_gap_grows(gaps: dict[str, float]) -> bool:
    ordered = [gaps[b] for b in ["tiny", "small", "base"] if b in gaps]
    if len(ordered) < 2:
        return True
    return all(ordered[i] >= ordered[i + 1] for i in range(len(ordered) - 1))


def decide(scrambled_table: dict, real_table: dict, model_name: str) -> dict:
    gaps_scrambled: dict[str, float] = {}
    gaps_real: dict[str, float] = {}

    for budget in BUDGET_ORDER:
        sc = scrambled_table.get(budget, {})
        re = real_table.get(budget, {})
        if "tamil" in sc and "latin" in sc:
            gaps_scrambled[budget] = sc["tamil"] - sc["latin"]
        if "tamil" in re and "latin" in re:
            gaps_real[budget] = re["tamil"] - re["latin"]

    budget_verdicts = {}
    for budget in BUDGET_ORDER:
        gs = gaps_scrambled.get(budget)
        gr = gaps_real.get(budget)
        if gs is None:
            budget_verdicts[budget] = "no_data"
            continue
        pct = (gs / gr) if (gr and gr > 0) else None
        budget_verdicts[budget] = {
            "scrambled_gap": round(gs, 4),
            "real_gap":      round(gr, 4) if gr is not None else None,
            "pct_of_real":   round(pct, 3) if pct is not None else None,
        }

    primary = "base" if "base" in gaps_scrambled else (
        max(gaps_scrambled, key=gaps_scrambled.get) if gaps_scrambled else None
    )
    if primary is None:
        return {"model": model_name, "verdict": "NO_DATA", "budget_verdicts": budget_verdicts}

    gs_primary = gaps_scrambled[primary]
    gr_primary = gaps_real.get(primary)
    pct = (gs_primary / gr_primary) if (gr_primary and gr_primary > 0) else None
    grows = monotone_gap_grows(gaps_scrambled)

    if gs_primary < 0.05 or (pct is not None and pct < 0.25):
        verdict = "NO-GO"
        reason  = (f"gap={gs_primary:.4f} (<0.05) OR pct_of_real={pct:.2%} (<25%) "
                   f"→ linguistic confound dominates, Pillar 3 not supported")
    elif gs_primary >= 0.10 and (pct is None or pct >= 0.50) and grows:
        verdict = "GO"
        reason  = (f"gap={gs_primary:.4f} (≥0.10) AND pct_of_real={pct:.2%} (≥50%) "
                   f"AND gap grows with tighter budget → visual confound confirmed")
    else:
        verdict = "REFINE"
        reason  = (f"gap={gs_primary:.4f}, pct_of_real={pct:.2%}, monotone={grows} "
                   f"→ borderline; check per-budget breakdown")

    return {
        "model":           model_name,
        "verdict":         verdict,
        "reason":          reason,
        "primary_budget":  primary,
        "gaps_scrambled":  {k: round(v, 4) for k, v in gaps_scrambled.items()},
        "gaps_real":       {k: round(v, 4) for k, v in gaps_real.items()},
        "gap_monotone":    grows,
        "budget_verdicts": budget_verdicts,
    }


def print_cer_table(cer_table: dict, title: str) -> None:
    scripts = ["tamil", "devanagari", "latin"]
    budgets = [b for b in BUDGET_ORDER if b in cer_table]
    print(f"\n{title}")
    hdr = f"  {'budget':>8} | " + " | ".join(f"{s:>12}" for s in scripts)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for b in budgets:
        row = cer_table[b]
        vals = [f"{row.get(s, float('nan')):>12.4f}" for s in scripts]
        print(f"  {b:>8} | " + " | ".join(vals))


def print_verdict(v: dict) -> None:
    print(f"\n{'='*60}")
    print(f"GATE A VERDICT [{v['model']}]: {v['verdict']}")
    print(f"Reason: {v['reason']}")
    print(f"Scrambled gaps (Tamil−Latin): {v.get('gaps_scrambled', {})}")
    print(f"Real gaps (Tamil−Latin):      {v.get('gaps_real', {})}")
    print(f"Gap grows tighter→worse:      {v.get('gap_monotone')}")
    print(f"{'='*60}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deepseek", required=True,
                    help="JSONL from infer_deepseek.py (all budgets)")
    ap.add_argument("--qwen",     default=None,
                    help="JSONL from infer_qwen.py (optional, all budgets)")
    ap.add_argument("--out",      default=None,
                    help="Write verdict JSON here")
    args = ap.parse_args()

    verdicts = []

    # DeepSeek Gate A
    ds_rows = load_results(Path(args.deepseek))
    ds_sc   = compute_cer_table(ds_rows, "scrambled")
    ds_re   = compute_cer_table(ds_rows, "real")
    print_cer_table(ds_sc, "DeepSeek-OCR-2 — Scrambled grapheme-CER by budget")
    print_cer_table(ds_re, "DeepSeek-OCR-2 — Real grapheme-CER by budget")
    ds_verdict = decide(ds_sc, ds_re, "DeepSeek-OCR-2")
    print_verdict(ds_verdict)
    verdicts.append(ds_verdict)

    # Qwen Gate A (optional)
    if args.qwen and Path(args.qwen).exists():
        qw_rows = load_results(Path(args.qwen))
        qw_sc   = compute_cer_table(qw_rows, "scrambled")
        qw_re   = compute_cer_table(qw_rows, "real")
        print_cer_table(qw_sc, "Qwen3-VL — Scrambled grapheme-CER by budget")
        print_cer_table(qw_re, "Qwen3-VL — Real grapheme-CER by budget")
        qw_verdict = decide(qw_sc, qw_re, "Qwen3-VL-2B")
        print_verdict(qw_verdict)
        verdicts.append(qw_verdict)

    # Primary verdict = DeepSeek (it ran all budgets for Gate A)
    primary = verdicts[0]
    output = {
        "gate_a_verdict":  primary["verdict"],
        "primary_model":   primary["model"],
        "reason":          primary["reason"],
        "all_models":      verdicts,
    }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nVerdict JSON → {args.out}")


if __name__ == "__main__":
    main()
