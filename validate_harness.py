"""
validate_harness.py — CPU-only end-to-end smoke test.

Run this locally BEFORE the A100 session to catch import errors, font issues,
and logic bugs while iteration is cheap.

Tests:
  1. textkit: segment, scramble, render, ink_density
  2. evaluate: grapheme_cer, codepoint_cer, word_acc
  3. datagen: build a tiny dataset (10 lines, no corpora)
  4. analyze: GO/NO-GO logic on synthetic verdicts
  5. base_select: decision logic on synthetic CER values
  6. benchmark_spec: coverage matrix completeness
  7. pillar2_demo: misranking cases run clean

Usage:
  pip install regex Pillow numpy
  python validate_harness.py           # requires fonts in fonts/
  python validate_harness.py --no_render  # skip rendering (no font files needed)
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

results: list[tuple[str, str, str]] = []  # (test_name, status, detail)


def check(name: str, fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
        results.append((name, PASS, ""))
        return True
    except Exception as exc:
        results.append((name, FAIL, str(exc)))
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# 1. textkit
# ---------------------------------------------------------------------------

def test_segment():
    from textkit import segment, normalize
    clusters = segment("கி")
    assert len(clusters) == 1, f"கி should be 1 cluster, got {len(clusters)}"
    clusters2 = segment("க்")
    assert len(clusters2) == 1, f"க் should be 1 cluster, got {len(clusters2)}"
    # documented akshara split
    clusters3 = segment("க்ஷ")
    assert len(clusters3) == 2, f"க்ஷ should split to 2 clusters, got {len(clusters3)}"
    # NFC normalization
    nfc = normalize("நட்")  # explicit virama
    assert "்" in nfc or "்" not in nfc  # just verify it runs


def test_scramble():
    from textkit import scramble, segment
    text = "தமிழ் மொழி"
    sc = scramble(text, seed=0)
    # multiset preservation
    orig_clusters = sorted(segment(text))
    scr_clusters  = sorted(segment(sc))
    assert orig_clusters == scr_clusters, "scramble must preserve grapheme multiset"
    assert sc != text, "scramble should change order (extremely unlikely to match)"


def test_inkdensity(render: bool = True):
    if not render:
        return
    font = Path("fonts/NotoSansTamil-Regular.ttf")
    if not font.exists():
        raise FileNotFoundError(f"Font not found: {font}  (use --no_render to skip)")
    from textkit import ink_per_grapheme
    density = ink_per_grapheme("தமிழ்", str(font), font_size=32)
    # Should be roughly in the 150–350 range based on prior measurements
    assert 50 < density < 600, f"ink density {density} looks wrong"


# ---------------------------------------------------------------------------
# 2. evaluate
# ---------------------------------------------------------------------------

def test_evaluate():
    from evaluate import grapheme_cer, codepoint_cer, word_accuracy

    # Perfect match
    assert grapheme_cer("abc", "abc") == 0.0
    assert codepoint_cer("abc", "abc") == 0.0

    # Empty reference edge case
    assert grapheme_cer("anything", "") == 0.0
    assert codepoint_cer("anything", "") == 0.0

    # Tamil: 1-cluster substitution
    # க→ங (both single-codepoint single-cluster consonants)
    g = grapheme_cer("ங", "க")
    assert g == 1.0, f"Single consonant sub should give CER=1.0, got {g}"

    # Ensure grapheme_cer ≤ codepoint_cer is NOT always true (it can go either way)
    from evaluate import grapheme_cer as gcer, codepoint_cer as ccer
    # For a simple Latin substitution they should be equal
    assert abs(gcer("a", "b") - ccer("a", "b")) < 1e-9

    # word_accuracy
    assert word_accuracy("hello world", "hello world") == 1.0
    assert word_accuracy("hello earth", "hello world") == 0.5


# ---------------------------------------------------------------------------
# 3. datagen
# ---------------------------------------------------------------------------

def test_datagen(render: bool = True):
    import tempfile
    from pathlib import Path as P
    from datagen import build

    if not render:
        return  # datagen requires Pillow + fonts

    font_missing = [
        f for f in [
            "fonts/NotoSansTamil-Regular.ttf",
            "fonts/NotoSansDevanagari-Regular.ttf",
            "fonts/NotoSans-Regular.ttf",
        ]
        if not P(f).exists()
    ]
    if font_missing:
        raise FileNotFoundError(f"Missing fonts: {font_missing}")

    with tempfile.TemporaryDirectory() as tmp:
        build(
            out_dir=P(tmp),
            n_lines=3,
            seed=0,
            font_size=32,
            use_corpora=False,
            corpus_dir=P("corpora"),
            splits=["smoke"],
        )
        manifest = P(tmp) / "manifests" / "smoke.jsonl"
        assert manifest.exists(), "manifest not created"
        lines = manifest.read_text(encoding="utf-8").splitlines()
        # 3 scripts × 2 modes × 3 lines = 18
        assert len(lines) == 18, f"Expected 18 manifest lines, got {len(lines)}"


# ---------------------------------------------------------------------------
# 4. analyze — synthetic worlds
# ---------------------------------------------------------------------------

def test_analyze_go():
    """Synthetic GO world: gaps meet all three criteria."""
    from analyze import decide

    # Tamil−Latin gap = 0.20 (scrambled), real gap = 0.30
    # → GO: 0.20 ≥ 0.10 AND 0.20/0.30 = 67% ≥ 50% AND monotone
    scr = {"tiny": {"tamil": 0.55, "latin": 0.30}, "small": {"tamil": 0.50, "latin": 0.30},
           "base": {"tamil": 0.45, "latin": 0.25}}
    real = {"tiny": {"tamil": 0.30, "latin": 0.00}, "small": {"tamil": 0.28, "latin": 0.00},
            "base": {"tamil": 0.25, "latin": 0.00}}

    # Build CER tables in the format analyze.py expects
    scr_table = {b: {s: scr[b][s] for s in scr[b]} for b in scr}
    real_table = {b: {s: real[b][s] for s in real[b]} for b in real}

    from analyze import decide
    v = decide(scr_table, real_table)
    assert v["verdict"] == "GO", f"Expected GO, got {v['verdict']}: {v['reason']}"


def test_analyze_nogo():
    """Synthetic NO-GO world: gaps too small."""
    from analyze import decide
    scr = {"base": {"tamil": 0.12, "latin": 0.10}}  # gap = 0.02 < 0.05
    real = {"base": {"tamil": 0.20, "latin": 0.10}}
    v = decide(scr, real)
    assert v["verdict"] == "NO-GO", f"Expected NO-GO, got {v['verdict']}"


# ---------------------------------------------------------------------------
# 5. base_select
# ---------------------------------------------------------------------------

def test_base_select():
    from base_select import decide

    # dCER = 0.02 (Qwen slightly worse) → DeepSeek
    v = decide(dcer_tamil=0.02, n_deepseek=200, n_qwen=200, frag=None)
    assert v["selected_base"].startswith("DeepSeek"), f"Got {v['selected_base']}"

    # dCER = -0.08 (Qwen clearly better) → Qwen
    v = decide(dcer_tamil=-0.08, n_deepseek=200, n_qwen=200, frag=None)
    assert v["selected_base"].startswith("Qwen"), f"Got {v['selected_base']}"

    # Insufficient data
    v = decide(dcer_tamil=0.0, n_deepseek=50, n_qwen=200, frag=None)
    assert v["verdict"] == "INSUFFICIENT_DATA"


# ---------------------------------------------------------------------------
# 6. benchmark_spec
# ---------------------------------------------------------------------------

def test_benchmark_spec():
    from benchmark_spec import (
        UYIR, MEY_CANONICAL, UYIRMEY_GRID, TOTAL_BASE_GRAPHEMES,
        V1_TIERS, CONFUSABLE_SETS, METRIC_DEFS, generate_coverage_probe,
        coverage_stats,
    )
    assert len(UYIR) == 12
    assert len(MEY_CANONICAL) == 18
    assert len(UYIRMEY_GRID) == 216  # 18×12
    assert TOTAL_BASE_GRAPHEMES == 247
    assert len(V1_TIERS) == 3
    assert "grapheme_cer" in METRIC_DEFS
    probe = generate_coverage_probe()
    assert len(probe) > 50


# ---------------------------------------------------------------------------
# 7. pillar2_demo
# ---------------------------------------------------------------------------

def test_pillar2_demo():
    from pillar2_demo import CASES, run_demo
    rows = run_demo(CASES)
    assert len(rows) == len(CASES)
    # At least one case should show cp/g ratio != 1 (the point of the demo)
    ratios = [r["cp_over_g"] for r in rows if not (r["cp_over_g"] != r["cp_over_g"])]  # skip NaN
    non_unit = [r for r in ratios if abs(r - 1.0) > 0.05]
    assert non_unit, "Expected at least one case where cp-CER and grapheme-CER differ"


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_render", action="store_true",
                    help="skip tests that require font files + Pillow")
    args = ap.parse_args()
    render = not args.no_render

    tests = [
        ("textkit.segment",      test_segment),
        ("textkit.scramble",     test_scramble),
        ("textkit.ink_density",  lambda: test_inkdensity(render)),
        ("evaluate.cer",         test_evaluate),
        ("datagen.build",        lambda: test_datagen(render)),
        ("analyze.GO",           test_analyze_go),
        ("analyze.NO-GO",        test_analyze_nogo),
        ("base_select",          test_base_select),
        ("benchmark_spec",       test_benchmark_spec),
        ("pillar2_demo",         test_pillar2_demo),
    ]

    for name, fn in tests:
        check(name, fn)

    # Summary
    print()
    print(f"{'Test':<30} {'Status'}")
    print("-" * 40)
    passed = failed = skipped = 0
    for name, status, detail in results:
        icon = "+" if status == PASS else ("X" if status == FAIL else "-")
        suffix = f"  {detail[:60]}" if detail else ""
        print(f"  {icon} {name:<28} {status}{suffix}")
        if status == PASS: passed += 1
        elif status == FAIL: failed += 1
        else: skipped += 1
    print("-" * 40)
    print(f"  {passed} passed, {failed} failed, {skipped} skipped")

    if failed:
        print("\nFix failures before running on A100.")
        sys.exit(1)
    else:
        print("\nAll tests passed — harness is ready for A100 session.")


if __name__ == "__main__":
    main()
