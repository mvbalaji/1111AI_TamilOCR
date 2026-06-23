# Tamil OCR Benchmark & Compression-Density Study

Harness for the paper: *Compression, Density, and Script: An Independent Benchmark
of 2026-Generation OCR-VLMs on Tamil* (arXiv, in preparation).

Three pillars:
1. **First independent open benchmark** of 2026 OCR-VLMs on Tamil
2. **Grapheme-aware evaluation protocol** (grapheme-cluster CER vs. codepoint-CER)
3. **Compression × script-density study** (Pillar 3 — subject to Gate A verdict)

Optional: **Path A+ Tamil-dedicated OCR-VLM** (if Gate A = GO).

---

## Quick start (local CPU — smoke test)

```bash
pip install -r requirements.txt
# Place Noto fonts in fonts/ (see CORPUS_PREP.md)
python textkit.py          # verify grapheme segmentation
python datagen.py          # build 200-line dataset from built-ins
python tokenizer_probe.py --model all   # fragmentation (needs HF access)
```

## Gate experiments (A100)

```bash
# 1. Populate corpora/ (see CORPUS_PREP.md) — or skip to use built-in lines
# 2. Run all gates in one session:
bash run_gates_a100.sh
# Outputs: results/pillar3_verdict.json  results/base_select_verdict.json
```

## File map

| File | Purpose |
|---|---|
| `textkit.py` | Grapheme seg, scrambler, renderer, ink-density |
| `datagen.py` | Build image dataset + JSONL manifests |
| `evaluate.py` | `grapheme_cer` + `codepoint_cer` (Pillar 2) |
| `tokenizer_probe.py` | Tokens/grapheme fragmentation (CPU) |
| `infer_deepseek.py` | DeepSeek-OCR inference, all budgets (A100) |
| `infer_qwen.py` | Qwen3-VL zero-shot inference (A100) |
| `analyze.py` | Gate A: Pillar-3 GO/NO-GO/REFINE decision |
| `base_select.py` | Gate B: base model selection |
| `run_gates_a100.sh` | One-shot A100 session script |
| `CORPUS_PREP.md` | How to populate `corpora/` and `fonts/` |

## Key empirical baselines (carry forward)

| Script | Ink/grapheme (Noto pt32) | DeepSeek tokens/grapheme | Qwen3-VL tokens/grapheme |
|---|---|---|---|
| Latin | ~117 | ~0.24 | ~0.24 |
| Devanagari | ~198 | ~1.19 | ~2.17 |
| Tamil | ~261 | ~0.96 | ~1.82 |

## Decision gates

**Gate A (Pillar 3 go/no-go):**
- GO: scrambled Tamil−Latin CER gap ≥ 0.10 AND ≥ 50% of real-text gap AND grows with tighter budget
- NO-GO: gap < 0.05 OR < 25% of real-text gap
- REFINE: between

**Gate B (base selection):**
- dCER = CER_Qwen − CER_DeepSeek on real Tamil (≥150 lines)
- dCER ≥ −0.03 → DeepSeek-OCR
- dCER ≤ −0.05 → Qwen3-VL
- Between → BORDERLINE (default DeepSeek for coherence)

## License

Code: Apache 2.0.
Fonts: OFL 1.1 (Google Noto).
Dataset (when released): CC BY 4.0.
