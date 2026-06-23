"""
evaluate.py — grapheme-cluster CER and codepoint CER (Pillar 2).

Core claim being tested:
  codepoint-CER misranks Tamil errors because Tamil matras and combining marks
  are multi-codepoint sequences that form a single perceptual unit.
  grapheme_cer measures edit distance over \\X clusters; codepoint_cer over
  individual Unicode codepoints.  We empirically show that a substitution of
  one matra can inflate codepoint-CER by 2–3× vs grapheme-CER.

Both metrics apply NFC normalization before comparison (canonical Pillar-2 form).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from textkit import normalize, segment


# ---------------------------------------------------------------------------
# Levenshtein edit distance (generic over sequences)
# ---------------------------------------------------------------------------

def _edit_distance(seq_a: list, seq_b: list) -> int:
    """Standard DP edit distance (insertion, deletion, substitution cost=1)."""
    m, n = len(seq_a), len(seq_b)
    # Use O(n) rolling row
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,      # insert
                prev[j] + 1,          # delete
                prev[j - 1] + cost,   # subst
            )
        prev = curr
    return prev[n]


# ---------------------------------------------------------------------------
# CER variants
# ---------------------------------------------------------------------------

def grapheme_cer(hypothesis: str, reference: str) -> float:
    """
    CER over Unicode grapheme clusters (\\X after NFC).
    This is the primary Pillar-2 metric.
    Returns edit_distance(clusters_hyp, clusters_ref) / len(clusters_ref).
    Returns 0.0 if reference is empty.
    """
    ref = segment(normalize(reference))
    hyp = segment(normalize(hypothesis))
    if not ref:
        return 0.0
    return _edit_distance(hyp, ref) / len(ref)


def codepoint_cer(hypothesis: str, reference: str) -> float:
    """
    CER over individual Unicode codepoints (after NFC).
    Included to demonstrate misranking on Tamil (Pillar 2 empirical demo).
    """
    ref = list(normalize(reference))
    hyp = list(normalize(hypothesis))
    if not ref:
        return 0.0
    return _edit_distance(hyp, ref) / len(ref)


def word_accuracy(hypothesis: str, reference: str) -> float:
    """
    Word-level accuracy (fraction of reference words that appear in hyp
    at the same position after whitespace tokenization).
    Low word-acc at low char-CER = segmentation/sandhi phenomenon noted in
    Jayatilleke & de Silva 2025 — we track this separately.
    """
    ref_words = normalize(reference).split()
    hyp_words = normalize(hypothesis).split()
    if not ref_words:
        return 1.0
    correct = sum(r == h for r, h in zip(ref_words, hyp_words))
    return correct / len(ref_words)


# ---------------------------------------------------------------------------
# Batch evaluation from manifest
# ---------------------------------------------------------------------------

def evaluate_manifest(
    manifest_path: Path,
    predictions_path: Path,
    out_path: Path | None = None,
) -> list[dict]:
    """
    manifest_path:    JSONL from datagen.py (fields: id, ground_truth, ...)
    predictions_path: JSONL with fields: id, prediction
    Returns list of result dicts with both CER metrics.
    """
    gt: dict[str, str] = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gt[rec["id"]] = rec["ground_truth"]

    preds: dict[str, str] = {}
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            preds[rec["id"]] = rec["prediction"]

    results = []
    for id_, reference in gt.items():
        hyp = preds.get(id_, "")
        results.append({
            "id": id_,
            "grapheme_cer": grapheme_cer(hyp, reference),
            "codepoint_cer": codepoint_cer(hyp, reference),
            "word_acc": word_accuracy(hyp, reference),
            "hypothesis": hyp,
            "reference": reference,
        })

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return results


def summary(results: list[dict]) -> dict:
    """Compute mean metrics across a result list."""
    if not results:
        return {}
    keys = ["grapheme_cer", "codepoint_cer", "word_acc"]
    return {k: sum(r[k] for r in results) / len(results) for k in keys}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",    required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out",         default=None)
    ap.add_argument("--by_script",   action="store_true",
                    help="break down summary by script field in manifest")
    args = ap.parse_args()

    results = evaluate_manifest(
        Path(args.manifest),
        Path(args.predictions),
        Path(args.out) if args.out else None,
    )
    s = summary(results)
    print(f"\nOverall  n={len(results)}")
    for k, v in s.items():
        print(f"  {k:20s}: {v:.4f}")

    if args.by_script:
        # reload manifest to get script field
        manifest_meta: dict[str, str] = {}
        with open(args.manifest, encoding="utf-8") as f:
            for ln in f:
                rec = json.loads(ln)
                manifest_meta[rec["id"]] = rec.get("script", "unknown")
        scripts: dict[str, list[dict]] = {}
        for r in results:
            sc = manifest_meta.get(r["id"], "unknown")
            scripts.setdefault(sc, []).append(r)
        for sc, recs in sorted(scripts.items()):
            s2 = summary(recs)
            print(f"\nScript={sc}  n={len(recs)}")
            for k, v in s2.items():
                print(f"  {k:20s}: {v:.4f}")


if __name__ == "__main__":
    main()
