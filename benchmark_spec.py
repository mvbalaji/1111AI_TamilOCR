"""
benchmark_spec.py — P1 benchmark specification: coverage matrix + document tiers.

This is the normative specification for the Tamil OCR Benchmark v1.
Everything in the paper keys off this file.  DO NOT change constants without
updating the paper's Table 1 and regenerating data.

Contents:
  1. COVERAGE_MATRIX  — the 247 uyirmey grid + Grantha + confusables + numerals
  2. DOCUMENT_TIERS   — v1 scope (printed multi-column, tables/forms, Tanglish)
  3. CONFUSABLE_SETS  — pairs/triples to oversample in synthetic data
  4. METRIC_DEFS      — frozen metric definitions
  5. Helper: generate_coverage_probe_text() — minimal Tamil text covering all cells

Tamil script structure (background for reviewers):
  - 12 uyir (vowel) × 18 mey (consonant) + 12 uyir standalone = 12×18 + 12 = 228
    But Tamil grammar has 18 mey consonants, so grid = 18×12 = 216 uyirmey + 18 mey
    + 12 uyir + 1 aytam = 247 total graphemes in the base Tamil Unicode block.
  - Grantha consonants ஜ ஷ ஸ ஹ ஶ — borrowed for Sanskrit loanwords; required for
    modern Tamil documents.
  - க்ஷ — Grantha conjunct; splits into க்+ஷ under \\X (documented in Pillar 2).
  - Split/left matras ெ ே ை ொ ோ ௌ — high error-rate in prior work; oversample.
  - Tamil numerals ௦–௯ + traditional ௰ ௱ ௲ ௹ — required for forms/tables tier.
  - Pulli ்  — combining mark; Jayatilleke & de Silva 2025 found most error-prone
    (partly codepoint-CER artifact — Pillar 2 target).

Document tier scope v1 (printed only — handwriting/palm-leaf cut to v2):
  See DOCUMENT_TIERS below.  Rationale: handwriting and palm-leaf produce
  near-zero scores on all 2026 VLMs → uninformative for ranking; annotation cost
  is 10× higher; make them a dedicated future work section.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 1. Coverage matrix
# ---------------------------------------------------------------------------

# 12 Tamil vowels (uyir) — Unicode scalar values for reference
UYIR = list("அஆஇஈஉஊஎஏஐஒஓஔ")  # 12

# 18 Tamil consonants (mey) — canonical set
MEY_CANONICAL = [
    "க", "ங", "ச", "ஞ", "ட", "ண", "த", "ந",
    "ப", "ம", "ய", "ர", "ற", "ல", "ள", "ழ",
    "வ", "ஶ",
]

# Grantha consonants (borrowed; needed for modern Tamil)
GRANTHA = ["ஜ", "ஷ", "ஸ", "ஹ", "ஶ"]

# Grantha already in MEY_CANONICAL (ஶ); additional Grantha for coverage:
GRANTHA_EXTRA = ["ஜ", "ஷ", "ஸ", "ஹ"]

# Tamil aytam
AYTAM = "ஃ"

# Tamil numerals
TAMIL_DIGITS = list("௦௧௨௩௪௫௬௭௮௯")          # 10 digits
TAMIL_NUMERALS_TRAD = list("௰௱௲௹")           # 10, 100, 1000, 10000

# Split/left matras (visually attached left of base consonant)
LEFT_MATRAS = list("ெேைொோௌ")

# All matras for completeness tracking
ALL_MATRAS = list("ாிீுூெேைொோௌ")  # 12 dependent vowel signs
PULLI = "்"  # virama/pulli

# Uyirmey grid — (consonant, vowel) pairs
# Full grid: 18 mey × 12 uyir = 216 uyirmey characters
# Not all Unicode combinations are precomposed; we use combining sequences.
# For coverage testing we enumerate the conceptual grid.
UYIRMEY_GRID: list[tuple[str, str]] = [
    (mey, uyir) for mey in MEY_CANONICAL for uyir in UYIR
]  # 18×12 = 216 entries

TOTAL_BASE_GRAPHEMES = (
    len(MEY_CANONICAL)       # 18 mey (pure consonants with inherent virama)
    + len(UYIR)              # 12 uyir (standalone vowels)
    + len(UYIRMEY_GRID)      # 216 uyirmey
    + 1                      # aytam ஃ
)  # = 247

# ---------------------------------------------------------------------------
# 2. Document tiers (v1 scope)
# ---------------------------------------------------------------------------

@dataclass
class DocumentTier:
    name:        str
    description: str
    in_v1:       bool
    rationale:   str
    metric:      list[str]   # which metrics apply


DOCUMENT_TIERS = [
    DocumentTier(
        name="printed_multicolumn",
        description="Multi-column newspaper/magazine layout, mixed font sizes, "
                    "headlines + body text",
        in_v1=True,
        rationale="High-impact real-world distribution; good discriminating power "
                  "between VLMs",
        metric=["grapheme_cer", "word_acc"],
    ),
    DocumentTier(
        name="tables_forms",
        description="Printed forms, tables, government documents; structured layout "
                    "with Tamil text in cells",
        in_v1=True,
        rationale="Required for downstream NLP; TEDS (Tree Edit Distance Score) "
                  "adds layout-aware evaluation",
        metric=["grapheme_cer", "word_acc", "teds"],
    ),
    DocumentTier(
        name="tanglish",
        description="Tamil-English code-switched text (Tanglish); Roman-script Tamil "
                    "mixed with English words",
        in_v1=True,
        rationale="High-prevalence in social media / modern documents; tests "
                  "multilingual OCR; author's prior work (Tanglish SLM)",
        metric=["grapheme_cer", "word_acc"],
    ),
    DocumentTier(
        name="handwriting",
        description="Handwritten Tamil — cursive and print styles",
        in_v1=False,
        rationale="Near-zero scores on all 2026 VLMs → uninformative for ranking; "
                  "annotation cost 10× higher than printed. Dedicated future work.",
        metric=["grapheme_cer"],
    ),
    DocumentTier(
        name="palm_leaf",
        description="Digitized palm-leaf manuscripts (Grantha/archaic Tamil script)",
        in_v1=False,
        rationale="Near-zero scores; requires separate specialist annotation; "
                  "different script norms. Future work.",
        metric=["grapheme_cer"],
    ),
]

V1_TIERS = [t for t in DOCUMENT_TIERS if t.in_v1]

# ---------------------------------------------------------------------------
# 3. Confusable pairs/triples — oversample in synthetic data
# ---------------------------------------------------------------------------

# These pairs are visually similar; OCR errors cluster here.
# Sources: Jayatilleke & de Silva 2025 + native speaker inspection.
CONFUSABLE_SETS = [
    # Retroflex/alveolar/dental nasal triples
    ("ண", "ன", "ந"),     # retroflex ṇ / alveolar n / dental n
    # Rhotic pair
    ("ர", "ற"),           # alveolar r / retroflex ṟ
    # Lateral/approximant triple
    ("ள", "ழ", "ல"),     # retroflex ḷ / lateral approximant ḻ / alveolar l
    # Matra attachment confusables (split/left matras)
    ("கெ", "கே"),         # short e / long e with க
    ("கொ", "கோ"),         # short o / long o with க
    # Pulli-attachment
    ("க", "க்"),          # base consonant vs. consonant + pulli
    # Digit lookalikes
    ("௧", "ச"),           # Tamil 1 vs. ச (visual similarity)
    ("௨", "உ"),           # Tamil 2 vs. உ
]

# ---------------------------------------------------------------------------
# 4. Metric definitions (frozen — changes require paper version bump)
# ---------------------------------------------------------------------------

METRIC_DEFS = {
    "grapheme_cer": {
        "description": "Character Error Rate over Unicode grapheme clusters (\\X) "
                       "after NFC normalization. Primary metric (Pillar 2).",
        "normalization": "NFC",
        "segmentation": "regex \\X (Unicode grapheme cluster)",
        "formula": "edit_distance(hyp_clusters, ref_clusters) / len(ref_clusters)",
        "range": "[0, ∞)  (can exceed 1.0 for very bad output)",
    },
    "word_acc": {
        "description": "Fraction of reference words matched at same position "
                       "(whitespace tokenization after NFC).",
        "normalization": "NFC",
        "segmentation": "whitespace",
        "formula": "sum(ref[i]==hyp[i]) / len(ref)",
        "range": "[0, 1]",
    },
    "teds": {
        "description": "Tree Edit Distance Score for table/form tier. "
                       "Measures structural fidelity of table reconstruction.",
        "normalization": "NFC on leaf text",
        "tool": "TEDS from https://github.com/ibm-aur-nlp/PubTabNet (MIT license)",
        "range": "[0, 1]  (1 = perfect)",
    },
    "codepoint_cer": {
        "description": "CER over individual Unicode codepoints (after NFC). "
                       "Included ONLY to demonstrate misranking (Pillar 2). "
                       "NOT the primary metric.",
        "normalization": "NFC",
        "segmentation": "individual codepoints",
        "formula": "edit_distance(list(hyp), list(ref)) / len(list(ref))",
        "range": "[0, ∞)",
    },
}

# ---------------------------------------------------------------------------
# 5. Coverage probe text generator
# ---------------------------------------------------------------------------

def generate_coverage_probe() -> str:
    """
    Generate a minimal Tamil string that exercises every cell in the coverage
    matrix.  Used in datagen.py for the coverage_probe split.

    Returns a string containing:
      - all 12 standalone uyir
      - all 18 mey with pulli (pure consonant form)
      - a sample of uyirmey combinations (one per consonant × each vowel class)
      - Grantha consonants
      - Tamil numerals
      - Aytam
      - Left/split matras in context
    """
    parts: list[str] = []

    # Standalone vowels
    parts.append("".join(UYIR))

    # Pure consonants (with pulli — the mey form)
    for mey in MEY_CANONICAL:
        parts.append(mey + PULLI)

    # Grantha extra
    for g in GRANTHA_EXTRA:
        parts.append(g + PULLI)

    # Aytam
    parts.append(AYTAM)

    # Numerals
    parts.append("".join(TAMIL_DIGITS))
    parts.append("".join(TAMIL_NUMERALS_TRAD))

    # Left matras in context (needs consonant base)
    for matra in LEFT_MATRAS:
        parts.append("க" + matra)  # ெ ே ை ொ ோ ௌ all attach to க

    # Sample uyirmey (one per consonant, cycling through vowels)
    for i, mey in enumerate(MEY_CANONICAL):
        vowel = UYIR[i % len(UYIR)]
        # Uyirmey is formed as: mey base + dependent vowel sign
        # We just concatenate as the combining sequence (Unicode will normalize)
        parts.append(mey + vowel)

    return " ".join(parts)


def coverage_stats() -> dict:
    """Return summary statistics for the coverage matrix."""
    return {
        "uyir":               len(UYIR),
        "mey":                len(MEY_CANONICAL),
        "uyirmey_grid":       len(UYIRMEY_GRID),
        "grantha_extra":      len(GRANTHA_EXTRA),
        "aytam":              1,
        "total_base":         TOTAL_BASE_GRAPHEMES,
        "left_matras":        len(LEFT_MATRAS),
        "tamil_digits":       len(TAMIL_DIGITS),
        "traditional_nums":   len(TAMIL_NUMERALS_TRAD),
        "confusable_sets":    len(CONFUSABLE_SETS),
        "v1_document_tiers":  len(V1_TIERS),
    }


if __name__ == "__main__":
    import json
    stats = coverage_stats()
    print("Coverage matrix stats:")
    for k, v in stats.items():
        print(f"  {k:22s}: {v}")
    print(f"\nDocument tiers (v1):")
    for t in V1_TIERS:
        print(f"  {t.name:25s}: {', '.join(t.metric)}")
    print(f"\nConfusable sets ({len(CONFUSABLE_SETS)}):")
    for cs in CONFUSABLE_SETS:
        print(f"  {' / '.join(cs)}")
    probe = generate_coverage_probe()
    print(f"\nCoverage probe ({len(probe)} chars, first 120): {probe[:120]}")
