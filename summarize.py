"""
summarize.py — Final multi-model comparison table for the paper.

Reads all model result JSONLs from a results directory and produces:
  1. Per-model grapheme-CER table (scripts × models)
  2. Pillar-3 budget × CER table for champion model
  3. Grapheme-CER vs codepoint-CER divergence table (Pillar 2)
  4. Summary JSON for hf_publish.py

Expected result files (auto-detected in --results_dir):
  gate_a_deepseek.jsonl     DeepSeek-OCR-2 (all budgets, real+scrambled)
  gate_a_qwen.jsonl         Qwen3-VL-2B    (all budgets, real+scrambled)  [optional]
  firered/gate.jsonl        FireRed-OCR    (real, base budget)             [optional]
  glmocr/gate.jsonl         GLM-OCR        (real, base budget)             [optional]
  paddleocr/gate.jsonl      PaddleOCR-VL   (real, base budget)             [optional]
  dotsocr/gate.jsonl        dots.ocr       (real, base budget)             [optional]

Usage:
  python summarize.py --results_dir results/ --out results/summary.json
  python summarize.py --results_dir results/ --champion deepseek --out results/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate import grapheme_cer, codepoint_cer

SCRIPTS = ["tamil", "devanagari", "latin"]
BUDGET_ORDER = ["tiny", "small", "base", "gundam"]


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def load_flat(path: Path, mode: str = "real",
              budget: str | None = None) -> dict[str, list[dict]]:
    """Load JSONL, filter by mode/budget. Returns {script: [rows]}."""
    by_script: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if mode and rec.get("mode") != mode:
                continue
            if budget and rec.get("budget") != budget:
                continue
            sc = rec.get("script", "unknown")
            by_script.setdefault(sc, []).append(rec)
    return by_script


def cer_row(by_script: dict[str, list[dict]]) -> dict[str, dict]:
    """Returns {script: {grapheme_cer, codepoint_cer, n}}."""
    result = {}
    for sc, rows in by_script.items():
        gcers, ccers = [], []
        for r in rows:
            pred = r.get("prediction", "")
            gt   = r.get("ground_truth", "")
            gcers.append(grapheme_cer(pred, gt))
            ccers.append(codepoint_cer(pred, gt))
        result[sc] = {
            "grapheme_cer":  round(_mean(gcers), 4),
            "codepoint_cer": round(_mean(ccers), 4),
            "n": len(rows),
        }
    return result


def detect_results(results_dir: Path) -> dict[str, Path]:
    """Auto-detect available result files."""
    candidates = {
        "DeepSeek-OCR-2":    results_dir / "gate_a_deepseek.jsonl",
        "Qwen3-VL-2B":       results_dir / "gate_a_qwen.jsonl",
        "1111AI_TamilOCR":   results_dir / "ft_eval_v2.jsonl",
        "FireRed-OCR":       results_dir / "firered"   / "gate.jsonl",
        "GLM-OCR":           results_dir / "glmocr"    / "gate.jsonl",
        "PaddleOCR-VL":      results_dir / "paddleocr" / "gate.jsonl",
        "dots.ocr":          results_dir / "dotsocr"   / "gate.jsonl",
        "Sarvam-Vision":     results_dir / "sarvam"    / "gate.jsonl",
        "Google-Cloud-Vision": results_dir / "gcloud"  / "gate.jsonl",
    }
    return {name: path for name, path in candidates.items() if path.exists()}


def print_model_table(scores: dict[str, dict[str, dict]]) -> None:
    """Print script × model grapheme-CER table."""
    models = list(scores.keys())
    print("\n=== Grapheme-CER by model and script (real text, base budget) ===")
    header = f"  {'script':>12} | " + " | ".join(f"{m:>16}" for m in models)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for sc in SCRIPTS:
        vals = []
        for m in models:
            v = scores[m].get(sc, {}).get("grapheme_cer", float("nan"))
            vals.append(f"{v:>16.4f}")
        print(f"  {sc:>12} | " + " | ".join(vals))


def print_pillar2_table(scores: dict[str, dict[str, dict]]) -> None:
    """Print grapheme vs codepoint CER divergence for Tamil."""
    print("\n=== Pillar 2: Grapheme-CER vs Codepoint-CER divergence (Tamil) ===")
    print(f"  {'model':>20} | {'grapheme_cer':>14} | {'codepoint_cer':>14} | {'ratio':>8}")
    print("  " + "-" * 68)
    for model, sc_data in scores.items():
        td = sc_data.get("tamil", {})
        gc = td.get("grapheme_cer", float("nan"))
        cc = td.get("codepoint_cer", float("nan"))
        ratio = cc / gc if gc > 0 else float("nan")
        print(f"  {model:>20} | {gc:>14.4f} | {cc:>14.4f} | {ratio:>8.2f}×")


def print_budget_table(results_dir: Path, champion: str) -> None:
    """Print champion model CER across budgets (Pillar 3)."""
    fname = "gate_a_deepseek.jsonl" if "deepseek" in champion.lower() else "gate_a_qwen.jsonl"
    path = results_dir / fname
    if not path.exists():
        return

    print(f"\n=== Pillar 3: {champion} — CER across budgets (Tamil, scrambled) ===")
    print(f"  {'budget':>8} | {'tamil_scr':>10} | {'latin_scr':>10} | {'gap':>8} | {'tamil_real':>10}")
    print("  " + "-" * 60)

    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]

    for budget in BUDGET_ORDER:
        scr_ta = [grapheme_cer(r["prediction"], r["ground_truth"])
                  for r in rows if r.get("budget") == budget
                  and r.get("mode") == "scrambled" and r.get("script") == "tamil"]
        scr_la = [grapheme_cer(r["prediction"], r["ground_truth"])
                  for r in rows if r.get("budget") == budget
                  and r.get("mode") == "scrambled" and r.get("script") == "latin"]
        re_ta  = [grapheme_cer(r["prediction"], r["ground_truth"])
                  for r in rows if r.get("budget") == budget
                  and r.get("mode") == "real" and r.get("script") == "tamil"]

        if not scr_ta:
            continue
        mst = _mean(scr_ta)
        msl = _mean(scr_la)
        mrt = _mean(re_ta) if re_ta else float("nan")
        print(f"  {budget:>8} | {mst:>10.4f} | {msl:>10.4f} | {mst-msl:>8.4f} | {mrt:>10.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/")
    ap.add_argument("--champion",    default="deepseek",
                    help="champion model name for Pillar 3 budget table")
    ap.add_argument("--budget",      default="base",
                    choices=BUDGET_ORDER,
                    help="which budget to use for cross-model comparison")
    ap.add_argument("--out",         default=None,
                    help="write JSON summary here (for hf_publish.py)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    available   = detect_results(results_dir)

    if not available:
        print(f"No result files found in {results_dir}. Run inference scripts first.")
        return

    print(f"Found results: {list(available.keys())}")

    # Compute per-model CER at base budget, real text
    scores: dict[str, dict[str, dict]] = {}
    for model_name, path in available.items():
        by_script = load_flat(path, mode="real", budget=args.budget)
        if not by_script:
            # Benchmark models may not have budget field
            by_script = load_flat(path, mode="real", budget=None)
        scores[model_name] = cer_row(by_script)

    print_model_table(scores)
    print_pillar2_table(scores)
    print_budget_table(results_dir, args.champion)

    # Summary JSON
    summary = {
        "budget_compared": args.budget,
        "mode": "real",
        "models": {
            model: {sc: data for sc, data in sc_data.items()}
            for model, sc_data in scores.items()
        },
    }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSummary JSON → {args.out}")


if __name__ == "__main__":
    main()
