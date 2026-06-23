"""
analyze.py — Pillar-3 GO / NO-GO / REFINE decision engine.

Decision criteria (from project spec, Section 6, Gate A):

  GO     — scrambled Tamil−Latin grapheme-CER gap ≥ 0.10
            AND ≥ 50% of the real-text Tamil−Latin gap
            AND gap grows monotonically as budget tightens
            (i.e., tiny > small > base in the gap metric)

  NO-GO  — gap < 0.05  OR  < 25% of real-text gap
            (indicates linguistic confound dominates, not visual)

  REFINE — everything else (borderline; dig into which budgets drive the gap)

Inputs:
  --deepseek_results  results/gate_a_deepseek.jsonl
      fields: id, base_id, script, mode, budget, grapheme_count, prediction, ground_truth

Output:
  Prints a decision table + verdict.  Optionally writes JSON summary.

Usage:
  python analyze.py --deepseek_results results/gate_a_deepseek.jsonl
  python analyze.py --deepseek_results results/gate_a_deepseek.jsonl --out results/pillar3_verdict.json
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


def compute_cer_table(
    rows: list[dict],
    mode: str,
) -> dict[str, dict[str, float]]:
    """
    Returns {budget: {script: mean_grapheme_cer}}.
    Only includes records with mode == mode.
    """
    # bucket[budget][script] = list of CER values
    bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        if row.get("mode") != mode:
            continue
        budget = row.get("budget", "base")
        script = row.get("script", "unknown")
        pred   = row.get("prediction", "")
        gt     = row.get("ground_truth", "")
        cer    = grapheme_cer(pred, gt)
        bucket[budget][script].append(cer)

    result: dict[str, dict[str, float]] = {}
    for budget, scripts in bucket.items():
        result[budget] = {sc: sum(v) / len(v) for sc, v in scripts.items() if v}
    return result


def monotone_gap_grows(gaps: dict[str, float]) -> bool:
    """
    Check if Tamil−Latin gap grows as we go tiny→small→base (tighter budget).
    We treat tiny as tightest.  Returns True if tiny >= small >= base (non-strict).
    """
    ordered = [gaps.get(b) for b in ["tiny", "small", "base"] if b in gaps]
    if len(ordered) < 2:
        return True  # not enough data to falsify
    return all(ordered[i] >= ordered[i + 1] for i in range(len(ordered) - 1))


def decide(
    scrambled_table: dict[str, dict[str, float]],
    real_table: dict[str, dict[str, float]],
) -> dict:
    """
    Run the GO/NO-GO/REFINE logic across budgets.
    Returns a summary dict with per-budget gaps and overall verdict.
    """
    gaps_scrambled: dict[str, float] = {}  # budget → tamil−latin gap (scrambled)
    gaps_real: dict[str, float] = {}       # budget → tamil−latin gap (real)

    for budget in BUDGET_ORDER:
        sc_row = scrambled_table.get(budget, {})
        re_row = real_table.get(budget, {})
        tamil_sc   = sc_row.get("tamil",  None)
        latin_sc   = sc_row.get("latin",  None)
        tamil_re   = re_row.get("tamil",  None)
        latin_re   = re_row.get("latin",  None)
        if tamil_sc is not None and latin_sc is not None:
            gaps_scrambled[budget] = tamil_sc - latin_sc
        if tamil_re is not None and latin_re is not None:
            gaps_real[budget] = tamil_re - latin_re

    budget_verdicts = {}
    for budget in BUDGET_ORDER:
        gs = gaps_scrambled.get(budget)
        gr = gaps_real.get(budget)
        if gs is None:
            budget_verdicts[budget] = "no_data"
            continue
        pct_of_real = (gs / gr) if (gr and gr > 0) else None
        budget_verdicts[budget] = {
            "scrambled_gap": round(gs, 4),
            "real_gap":      round(gr, 4) if gr is not None else None,
            "pct_of_real":   round(pct_of_real, 3) if pct_of_real is not None else None,
        }

    # Aggregate across available budgets (use base as primary if present, else max)
    primary = "base" if "base" in gaps_scrambled else (
        max(gaps_scrambled, key=gaps_scrambled.get) if gaps_scrambled else None
    )
    if primary is None:
        return {"verdict": "NO_DATA", "budget_verdicts": budget_verdicts}

    gs_primary = gaps_scrambled[primary]
    gr_primary = gaps_real.get(primary)
    pct = (gs_primary / gr_primary) if (gr_primary and gr_primary > 0) else None

    grows = monotone_gap_grows(gaps_scrambled)

    # Decision logic
    if gs_primary < 0.05 or (pct is not None and pct < 0.25):
        verdict = "NO-GO"
        reason = (f"gap={gs_primary:.4f} (<0.05) OR pct_of_real={pct:.2%} (<25%) "
                  f"→ linguistic confound dominates, Pillar 3 not supported")
    elif gs_primary >= 0.10 and (pct is None or pct >= 0.50) and grows:
        verdict = "GO"
        reason = (f"gap={gs_primary:.4f} (≥0.10) AND pct_of_real={pct:.2%} (≥50%) "
                  f"AND gap grows with tighter budget → visual confound confirmed")
    else:
        verdict = "REFINE"
        reason = (f"gap={gs_primary:.4f}, pct_of_real={pct:.2%}, monotone={grows} "
                  f"→ borderline; check per-budget breakdown and add more data")

    return {
        "verdict":         verdict,
        "reason":          reason,
        "primary_budget":  primary,
        "gaps_scrambled":  {k: round(v, 4) for k, v in gaps_scrambled.items()},
        "gaps_real":       {k: round(v, 4) for k, v in gaps_real.items()},
        "gap_monotone":    grows,
        "budget_verdicts": budget_verdicts,
    }


def print_table(cer_table: dict[str, dict[str, float]], title: str) -> None:
    scripts = ["tamil", "devanagari", "latin"]
    budgets = [b for b in BUDGET_ORDER if b in cer_table]
    print(f"\n{title}")
    header = f"  {'budget':>8} | " + " | ".join(f"{s:>12}" for s in scripts)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for b in budgets:
        row = cer_table[b]
        vals = [f"{row.get(s, float('nan')):>12.4f}" for s in scripts]
        print(f"  {b:>8} | " + " | ".join(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deepseek_results", required=True,
                    help="JSONL from infer_deepseek.py")
    ap.add_argument("--out", default=None,
                    help="write JSON verdict summary here")
    args = ap.parse_args()

    rows = load_results(Path(args.deepseek_results))
    scrambled_table = compute_cer_table(rows, mode="scrambled")
    real_table      = compute_cer_table(rows, mode="real")

    print_table(scrambled_table, "Scrambled-mode grapheme-CER by budget and script")
    print_table(real_table,      "Real-mode grapheme-CER by budget and script")

    verdict = decide(scrambled_table, real_table)

    print(f"\n{'='*60}")
    print(f"PILLAR-3 VERDICT: {verdict['verdict']}")
    print(f"Reason: {verdict['reason']}")
    print(f"Gap by budget (Tamil−Latin, scrambled): {verdict['gaps_scrambled']}")
    print(f"Gap by budget (Tamil−Latin, real):      {verdict['gaps_real']}")
    print(f"Gap grows with tighter budget:           {verdict['gap_monotone']}")
    print(f"{'='*60}\n")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(verdict, f, indent=2, ensure_ascii=False)
        print(f"Verdict JSON: {args.out}")


if __name__ == "__main__":
    main()
