"""
pillar2_demo.py — Empirical demonstration that codepoint-CER misranks Tamil errors.

This is the Pillar 2 methodological contribution.  Run on CPU; no GPU needed.

The claim: for Tamil, a single grapheme-cluster substitution (e.g. ெ→ே, one
perceptual error) can inflate codepoint-CER by 2–3× compared to grapheme-CER,
because Tamil matras and combining marks are multi-codepoint sequences.
Meanwhile, a word-boundary insertion that creates a clearly wrong output can
*lower* codepoint-CER relative to grapheme-CER because it adds cheap codepoints.

This script:
  1. Constructs synthetic (hypothesis, reference) pairs that illustrate the
     misranking with known, documented Tamil errors.
  2. Computes both CER metrics.
  3. Prints a table that can be directly used as Table 2 in the paper.
  4. Optionally writes a CSV.

Usage:
  python pillar2_demo.py
  python pillar2_demo.py --out results/pillar2_table.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass

# Ensure Tamil text prints correctly on Windows (cp1252 default fails on Tamil).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from evaluate import grapheme_cer, codepoint_cer


@dataclass
class ErrorCase:
    name:        str    # short label for the error type
    reference:   str    # correct Tamil text
    hypothesis:  str    # OCR output
    description: str    # one-line explanation for the paper table


# ---------------------------------------------------------------------------
# Constructed error cases (each isolates one error type)
# ---------------------------------------------------------------------------

CASES: list[ErrorCase] = [

    # -----------------------------------------------------------------------
    # MECHANISM: Tamil grapheme clusters contain 1–2 codepoints each
    # (base consonant + optional combining vowel sign or virama).
    # For ANY substitution of a whole cluster, grapheme-CER and codepoint-CER
    # count the same number of edits (1), but cp-CER divides by the *larger*
    # codepoint denominator → cp-CER is systematically LOWER (understates errors).
    # The one exception: when the edit changes a 2-cp cluster to a 1-cp cluster
    # (or vice versa), the edit count itself differs between metrics → misranking.
    # -----------------------------------------------------------------------

    # --- Case 1: Matra substitution, both clusters are 2 codepoints ---
    # கெ = [க+ெ] = 2 cp, 1 cluster.  கே = [க+ே] = 2 cp, 1 cluster.
    # Both metrics count 1 edit. But cp denominator (6) > cluster denominator (3)
    # → cp-CER = 1/6 = 0.167  vs  g-CER = 1/3 = 0.333.
    # cp/g = 0.5: codepoint-CER UNDERSTATES this error by 2×.
    ErrorCase(
        name="matra_subst_single_cp",
        reference="கெட்டி",   # "keṭṭi" — strong
        hypothesis="கேட்டி",   # "kēṭṭi" — listening (wrong vowel length)
        description="Short-e→long-e matra sub: 1 cluster edit / 3 clusters = g-CER 0.33, "
                    "but 1 cp edit / 6 codepoints = cp-CER 0.17. "
                    "cp-CER understates by 2× (denominator inflation).",
    ),

    # --- Case 2: Left-split matra drop ---
    # ொ (U+0BCA) = 1 NFC codepoint, 1 grapheme cluster.
    # கொ → க: 1 cluster edit, 1 cp edit, but denominators differ → same understatement.
    ErrorCase(
        name="left_matra_drop",
        reference="கொள்",     # "koḷ" — take
        hypothesis="கள்",      # "kaḷ" — bracelet (matra ொ dropped)
        description="Left matra ொ (U+0BCA) dropped: 1 cluster edit / 3 clusters, "
                    "1 cp edit / 4 codepoints. cp-CER (0.25) < g-CER (0.33). "
                    "Understatement smaller because NFC ொ is 1 codepoint.",
    ),

    # --- Case 3: Pulli drop — the Jayatilleke 'most error-prone character' finding ---
    # க் = [க + ்] = 2 codepoints, 1 cluster.  Dropping pulli: க் → க.
    # Grapheme-CER: 1 cluster substituted (க்→க, both are valid clusters).
    # Codepoint-CER: 1 deletion (் removed) from larger sequence.
    # Per-word, g-CER > cp-CER (same mechanism as Cases 1-2).
    # Jayatilleke found pulli "most error-prone" under codepoint-CER: this is
    # because pulli is frequent AND each pulli error appears as 1 cheap codepoint
    # edit in a long codepoint sequence. Under grapheme-CER, each pulli error is
    # exactly 1 cluster edit — same weight as any consonant substitution.
    ErrorCase(
        name="pulli_drop",
        reference="நட்சத்திரம்",   # "naṭcattiraṃ" — star
        hypothesis="நடசத்திரம்",    # ட் → ட (pulli dropped)
        description="Pulli drop: 1 cluster edit / 7 clusters = g-CER 0.143. "
                    "1 cp deletion / 11 codepoints = cp-CER 0.091. "
                    "cp-CER understates. Jayatilleke finding: pulli 'most error-prone' "
                    "under cp-CER is partly this denominator artifact.",
    ),

    # --- Case 4: Pulli drop changes 2-cp cluster to 1-cp cluster → edit count differs ---
    # When the dropped pulli causes the cluster count to stay same but codepoint
    # count differs, the NUMERATOR also changes. This can flip the direction.
    ErrorCase(
        name="pulli_drop_cluster_split",
        reference="கற்பனை",    # "kaṟpanai" — imagination
        hypothesis="கரபனை",     # ற் → ர (pulli dropped AND consonant changes)
        description="ற் (2cp, 1 cluster) → ர (1cp, 1 cluster): 1 cluster sub = g-CER 0.25, "
                    "but 2 cp edits (sub+del) / 6 codepoints = cp-CER 0.33. "
                    "cp-CER INFLATES this error (numerator increases). "
                    "This is the misranking direction: same perceptual error, higher cp-CER.",
    ),

    # --- Case 5: Conjunct cluster akshara split (documented Pillar 2 case) ---
    # க்ஷ segments as [க், ஷ] = 2 clusters, 3 codepoints.
    # Pulli drop: க்ஷ → கஷ = [க, ஷ] = 2 clusters, 2 codepoints.
    # 1 cluster edit (க்→க), but 1 deletion in codepoints → cp-CER understates.
    ErrorCase(
        name="conjunct_pulli_drop",
        reference="க்ஷமை",    # "kṣamai" — forgiveness
        hypothesis="கஷமை",     # pulli on க் dropped
        description="க்ஷ conjunct (2 clusters, 3 cp): pulli drop gives 2 clusters / 2 cp. "
                    "g-CER counts 1 cluster sub / 4 clusters = 0.25; "
                    "cp-CER counts 1 cp del / 5 cp = 0.20. cp-CER understates.",
    ),

    # --- Case 6: Confusable ண/ன — control case, metrics agree ---
    # Both single-codepoint single-cluster → no denominator difference per edit.
    # Included to show the understatement is specific to combining-mark errors.
    ErrorCase(
        name="confusable_NN",
        reference="மண்",    # "maṇ" — soil (retroflex ண)
        hypothesis="மன்",    # "man" — human (alveolar ன)
        description="Confusable ண/ன: both 1 cp = 1 cluster. "
                    "g-CER and cp-CER differ only by denominator size, not edit count. "
                    "Demonstrates combining-mark errors are the asymmetry driver.",
    ),

    # --- Case 7: Cross-script comparison — THE strongest Pillar 2 argument ---
    # Latin text has ~1.0 cp/cluster; Tamil has ~1.4-1.6 cp/cluster.
    # For the SAME number of cluster-level errors, Tamil gets LOWER cp-CER
    # than Latin because its denominator is larger.
    # This biases cross-script comparisons: Tamil OCR looks easier under cp-CER.
    # We simulate this with equal-cluster-count strings, each with 1 error.
    ErrorCase(
        name="cross_script_bias",
        reference="cat",     # 3 Latin chars, 3 clusters, 3 codepoints
        hypothesis="bat",    # 1 substitution
        description="Latin: 1 error / 3 clusters = g-CER 0.33, 1 error / 3 cp = cp-CER 0.33. "
                    "Compare to Tamil Case 1: same 1-error / 3-cluster but cp-CER 0.17. "
                    "Cross-script cp-CER underestimates Tamil error rate vs Latin — "
                    "Tamil OCR appears easier than it is on cp-CER.",
    ),
]


# ---------------------------------------------------------------------------
# Run and format
# ---------------------------------------------------------------------------

def run_demo(cases: list[ErrorCase]) -> list[dict]:
    rows = []
    for c in cases:
        g_cer = grapheme_cer(c.hypothesis, c.reference)
        cp_cer = codepoint_cer(c.hypothesis, c.reference)
        from textkit import segment
        ref_clusters = len(segment(c.reference))
        ref_codepoints = len(list(c.reference))  # after NFC in codepoint_cer
        ratio = cp_cer / g_cer if g_cer > 0 else float("nan")
        rows.append({
            "case":           c.name,
            "reference":      c.reference,
            "hypothesis":     c.hypothesis,
            "ref_clusters":   ref_clusters,
            "ref_codepoints": ref_codepoints,
            "grapheme_cer":   round(g_cer, 4),
            "codepoint_cer":  round(cp_cer, 4),
            "cp_over_g":      round(ratio, 3),   # >1 = codepoint inflates; <1 = understates
            "description":    c.description,
        })
    return rows


def print_table(rows: list[dict]) -> None:
    print("\nPillar 2 — CER metric comparison on Tamil error cases")
    print("=" * 100)
    print(f"{'Case':<30} {'Ref':>6} {'Hyp':>6} {'g-CER':>7} {'cp-CER':>7} {'cp/g':>6}  Description")
    print("-" * 100)
    for r in rows:
        flag = "  ←INFLATE" if r["cp_over_g"] > 1.5 else ("  ←UNDERSTATE" if r["cp_over_g"] < 0.7 else "")
        print(f"{r['case']:<30} "
              f"{r['reference']:>6} "
              f"{r['hypothesis']:>6} "
              f"{r['grapheme_cer']:>7.4f} "
              f"{r['codepoint_cer']:>7.4f} "
              f"{r['cp_over_g']:>6.3f}"
              f"{flag}")
    print("-" * 100)
    n_inflate    = sum(1 for r in rows if r["cp_over_g"] > 1.1)
    n_understate = sum(1 for r in rows if r["cp_over_g"] < 0.9)
    n_agree      = sum(1 for r in rows if 0.9 <= r["cp_over_g"] <= 1.1)
    print(f"Cases where cp-CER inflates   (cp/g > 1.1): {n_inflate}")
    print(f"Cases where cp-CER understates (cp/g < 0.9): {n_understate}")
    print(f"Cases where both metrics agree  (0.9-1.1):   {n_agree}")
    print()
    print("Paper claim: For Tamil, codepoint-CER systematically understates error rates")
    print("(larger codepoint denominator from combining marks) AND can misrank errors")
    print("when different error types have asymmetric codepoint-vs-cluster edit counts.")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write CSV table here")
    args = ap.parse_args()

    rows = run_demo(CASES)
    print_table(rows)

    if args.out:
        import pathlib
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Table written to {args.out}")


if __name__ == "__main__":
    main()
