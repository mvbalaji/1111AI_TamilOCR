#!/usr/bin/env bash
# run_gates_a100.sh — single A100 session: Gate A (Pillar-3) + Gate B (base selection)
#                     + full P1 benchmark run (all 6 models).
#
# Prerequisites (do these BEFORE this script):
#   1. pip install git+https://github.com/huggingface/transformers
#      pip install flash-attn --no-build-isolation
#      pip install regex Pillow numpy datasets accelerate sentencepiece protobuf
#      pip install pytesseract   # for Tesseract baseline (apt install tesseract-ocr tesseract-ocr-tam)
#   2. python download_fonts.py   # downloads 10 fonts including 7 Tamil variants
#   3. huggingface-cli login       # HF credentials for gated models
#   4. python validate_harness.py  # all 10 tests must pass
#
# Model slugs used:
#   Gate A+B: deepseek-ai/DeepSeek-OCR-2, Qwen/Qwen3-VL-2B-Instruct
#   P1 benchmark: + FireRedTeam/FireRed-OCR, zai-org/GLM-OCR,
#                   PaddlePaddle/PaddleOCR-VL-1.6, rednote-hilab/dots.ocr
#   CPU baseline: Tesseract 5 (tam/hin/eng)
#
# Estimated A100 runtime (80GB):
#   corpus_prep     ~10 min
#   datagen         ~8  min (multi-font + augmentation)
#   tokenizer_probe ~5  min
#   deepseek        ~60 min  (4 budgets × 2 modes × 600 images)
#   qwen            ~30 min
#   firered         ~25 min
#   glmocr          ~20 min
#   paddleocr       ~20 min
#   dotsocr         ~20 min
#   tesseract       ~5  min  (CPU-parallel)
#   analysis        <1  min

set -euo pipefail
mkdir -p results

N_LINES=200
SEED_1=42
SEED_2=7

DEEPSEEK_ID="deepseek-ai/DeepSeek-OCR-2"
QWEN_ID="Qwen/Qwen3-VL-2B-Instruct"
FIRERED_ID="FireRedTeam/FireRed-OCR"
GLMOCR_ID="zai-org/GLM-OCR"
PADDLE_ID="PaddlePaddle/PaddleOCR-VL-1.6"
DOTS_ID="rednote-hilab/dots.ocr"

echo "================================================================"
echo "Step 1: Corpus preparation"
echo "================================================================"
python corpus_prep.py --n 500

echo ""
echo "================================================================"
echo "Step 2: Generate dataset — seed ${SEED_1}, multi-font, medium augmentation"
echo "================================================================"
python datagen.py \
    --out_dir data \
    --corpus_dir corpora \
    --n_lines $N_LINES \
    --seed $SEED_1 \
    --font_size 32 \
    --splits gate \
    --use_corpora \
    --multi_font \
    --aug_level medium \
    --oversample_confusable

# Also generate a clean split for P2 confusable analysis (no augmentation)
python datagen.py \
    --out_dir data \
    --corpus_dir corpora \
    --n_lines $N_LINES \
    --seed $SEED_1 \
    --font_size 32 \
    --splits gate_clean \
    --use_corpora

echo ""
echo "================================================================"
echo "Step 3: Tokenizer fragmentation probe (CPU — no GPU needed)"
echo "================================================================"
python tokenizer_probe.py --model all \
    2>&1 | tee results/tokenizer_fragmentation.txt

echo ""
echo "================================================================"
echo "Step 4: Tesseract baseline — CPU, runs now while GPU warms up"
echo "================================================================"
python infer_tesseract.py data/manifests/gate.jsonl \
    2>&1 | tee results/tesseract_run.log

echo ""
echo "================================================================"
echo "Step 5: DeepSeek-OCR-2 — all budgets, both modes  (Gate A + Gate B)"
echo "================================================================"
python infer_deepseek.py \
    --manifest data/manifests/gate.jsonl \
    --out results/gate_a_deepseek.jsonl \
    --budget tiny small base gundam \
    --model_id "$DEEPSEEK_ID"

echo ""
echo "================================================================"
echo "Step 6: Qwen3-VL — real mode only  (Gate B)"
echo "================================================================"
python infer_qwen.py \
    --manifest data/manifests/gate.jsonl \
    --out results/gate_b_qwen.jsonl \
    --mode real \
    --model_id "$QWEN_ID"

echo ""
echo "================================================================"
echo "Step 7: Gate A — Pillar-3 GO / NO-GO / REFINE"
echo "================================================================"
python analyze.py \
    --deepseek_results results/gate_a_deepseek.jsonl \
    --out results/pillar3_verdict.json

echo ""
echo "================================================================"
echo "Step 8: Gate B — base model selection"
echo "================================================================"
python base_select.py \
    --deepseek_results results/gate_a_deepseek.jsonl \
    --qwen_results     results/gate_b_qwen.jsonl \
    --out results/base_select_verdict.json

echo ""
echo "================================================================"
echo "Step 9: P1 Benchmark — FireRed-OCR"
echo "================================================================"
python infer_firered.py data/manifests/gate.jsonl \
    2>&1 | tee results/firered_run.log

echo ""
echo "================================================================"
echo "Step 10: P1 Benchmark — GLM-OCR"
echo "================================================================"
python infer_glmocr.py data/manifests/gate.jsonl \
    2>&1 | tee results/glmocr_run.log

echo ""
echo "================================================================"
echo "Step 11: P1 Benchmark — PaddleOCR-VL-1.6"
echo "================================================================"
python infer_paddleocr.py data/manifests/gate.jsonl \
    2>&1 | tee results/paddleocr_run.log

echo ""
echo "================================================================"
echo "Step 12: P1 Benchmark — dots.ocr"
echo "================================================================"
python infer_dotsocr.py data/manifests/gate.jsonl \
    2>&1 | tee results/dotsocr_run.log

echo ""
echo "================================================================"
echo "Step 13: Evaluate all models — grapheme-CER + codepoint-CER"
echo "================================================================"
for MODEL_TAG in deepseek qwen firered glmocr paddleocr dotsocr tesseract; do
    RESULTS_JSONL=$(find results/${MODEL_TAG} -name "gate.jsonl" 2>/dev/null | head -1 || \
                    echo "results/${MODEL_TAG}/gate.jsonl")
    if [ -f "$RESULTS_JSONL" ]; then
        python evaluate.py --manifest "$RESULTS_JSONL" \
            --out "results/eval_${MODEL_TAG}.json" \
            2>&1 | tee "results/eval_${MODEL_TAG}.log"
    else
        echo "  SKIP $MODEL_TAG — results not found at $RESULTS_JSONL"
    fi
done

echo ""
echo "================================================================"
echo "Step 14: Second seed for Gate B robustness (optional)"
echo "================================================================"
echo "# Uncomment to run second seed:"
echo "# python datagen.py --out_dir data_s2 --corpus_dir corpora \\"
echo "#     --n_lines $N_LINES --seed $SEED_2 --font_size 32 --splits gate --use_corpora"
echo "# python infer_deepseek.py --manifest data_s2/manifests/gate.jsonl \\"
echo "#     --out results/gate_b_deepseek_s2.jsonl --mode real --budget base"
echo "# python infer_qwen.py --manifest data_s2/manifests/gate.jsonl \\"
echo "#     --out results/gate_b_qwen_s2.jsonl --mode real"
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
echo "DONE. Check eval_*.json for per-model benchmark scores."
echo "Next: python hf_publish.py dataset --dry_run"
