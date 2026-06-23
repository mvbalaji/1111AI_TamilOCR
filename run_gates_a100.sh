#!/usr/bin/env bash
# run_gates_a100.sh — single A100 session: Gate A (Pillar-3) + Gate B (base selection).
#
# Prerequisites (do these BEFORE this script):
#   1. pip install git+https://github.com/huggingface/transformers
#      pip install flash-attn --no-build-isolation
#      pip install regex Pillow numpy datasets accelerate sentencepiece protobuf
#   2. Fonts in fonts/:
#        fonts/NotoSansTamil-Regular.ttf
#        fonts/NotoSansDevanagari-Regular.ttf
#        fonts/NotoSans-Regular.ttf
#   3. HF credentials: huggingface-cli login
#      (verify deepseek-ai/DeepSeek-OCR-2 is accessible — check if gated)
#   4. Local smoke-test passed:
#        python validate_harness.py --no_render   # no fonts needed
#        python validate_harness.py               # with fonts
#
# Expected A100 runtime (80GB, all estimates — update after first run):
#   corpus_prep   ~10 min (streaming download)
#   datagen       ~5  min (200 lines × 3 scripts × 2 modes)
#   deepseek      ~60 min (200 × 3 × 2 × 4 = 4800 calls, ~0.75s each)
#   qwen          ~30 min (200 × 3 × 1 mode = 600 calls)
#   analysis      <1  min
#
# Outputs:
#   results/gate_a_deepseek.jsonl     — all budgets, both modes, all scripts
#   results/gate_b_qwen.jsonl         — real mode only
#   results/pillar3_verdict.json      — Gate A decision (GO / NO-GO / REFINE)
#   results/base_select_verdict.json  — Gate B decision (DeepSeek / Qwen / BORDERLINE)

set -euo pipefail
mkdir -p results

N_LINES=200   # ≥150 required; 200 for headroom and 2-seed robustness
SEED_1=42
SEED_2=7      # second seed for Gate B trustworthiness check

DEEPSEEK_ID="deepseek-ai/DeepSeek-OCR-2"   # verify at hf.co/deepseek-ai
QWEN_ID="Qwen/Qwen3-VL-2B-Instruct"        # verify at hf.co/Qwen

echo "================================================================"
echo "Step 1: Corpus preparation"
echo "================================================================"
python corpus_prep.py --n 500

echo ""
echo "================================================================"
echo "Step 2: Generate dataset — seed ${SEED_1}"
echo "================================================================"
python datagen.py \
    --out_dir data \
    --corpus_dir corpora \
    --n_lines $N_LINES \
    --seed $SEED_1 \
    --font_size 32 \
    --splits gate \
    --use_corpora

echo ""
echo "================================================================"
echo "Step 3: Tokenizer fragmentation probe (CPU — no GPU needed)"
echo "================================================================"
python tokenizer_probe.py --model all \
    2>&1 | tee results/tokenizer_fragmentation.txt

echo ""
echo "================================================================"
echo "Step 4: DeepSeek-OCR-2 — all budgets, both modes  (Gate A + Gate B half)"
echo "================================================================"
# Running both modes in one pass to avoid loading the model twice.
# Gate A reads: scrambled rows
# Gate B reads: real rows (base budget only needed, but all budgets free here)
python infer_deepseek.py \
    --manifest data/manifests/gate.jsonl \
    --out results/gate_a_deepseek.jsonl \
    --budget tiny small base gundam \
    --model_id "$DEEPSEEK_ID"

echo ""
echo "================================================================"
echo "Step 5: Qwen3-VL — real mode only  (Gate B)"
echo "================================================================"
python infer_qwen.py \
    --manifest data/manifests/gate.jsonl \
    --out results/gate_b_qwen.jsonl \
    --mode real \
    --model_id "$QWEN_ID"

echo ""
echo "================================================================"
echo "Step 6: Gate A — Pillar-3 GO / NO-GO / REFINE"
echo "================================================================"
python analyze.py \
    --deepseek_results results/gate_a_deepseek.jsonl \
    --out results/pillar3_verdict.json

echo ""
echo "================================================================"
echo "Step 7: Gate B — base model selection"
echo "================================================================"
python base_select.py \
    --deepseek_results results/gate_a_deepseek.jsonl \
    --qwen_results     results/gate_b_qwen.jsonl \
    --out results/base_select_verdict.json

echo ""
echo "================================================================"
echo "Optional Step 8: Second seed for Gate B robustness"
echo "================================================================"
echo "# Uncomment to run second seed (recommended before trusting Gate B verdict):"
echo "# python datagen.py --out_dir data_s2 --corpus_dir corpora \\"
echo "#     --n_lines $N_LINES --seed $SEED_2 --font_size 32 --splits gate --use_corpora"
echo "# python infer_deepseek.py --manifest data_s2/manifests/gate.jsonl \\"
echo "#     --out results/gate_b_deepseek_s2.jsonl --mode real --budget base --model_id $DEEPSEEK_ID"
echo "# python infer_qwen.py --manifest data_s2/manifests/gate.jsonl \\"
echo "#     --out results/gate_b_qwen_s2.jsonl --mode real --model_id $QWEN_ID"
echo "# python base_select.py --deepseek_results results/gate_b_deepseek_s2.jsonl \\"
echo "#     --qwen_results results/gate_b_qwen_s2.jsonl --out results/base_select_verdict_s2.json"

echo ""
echo "================================================================"
echo "GATE RESULTS"
echo "================================================================"
echo "--- Gate A (Pillar 3 go/no-go) ---"
cat results/pillar3_verdict.json
echo ""
echo "--- Gate B (base model selection) ---"
cat results/base_select_verdict.json
echo ""
echo "DONE. Report both verdicts before proceeding to P1/P4."
