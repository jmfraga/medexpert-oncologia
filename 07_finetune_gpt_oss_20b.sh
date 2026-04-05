#!/bin/bash
# Fine-tune gpt-oss-20b (4-bit) for MedExpert Oncology
# Dataset: 142K clean (data/clean/)
# Ejecutar en M4 (64GB RAM, Apple Silicon)
#
# Uso:
#   bash 07_finetune_gpt_oss_20b.sh              # Run completo (3000 iters)
#   bash 07_finetune_gpt_oss_20b.sh --pilot      # Run piloto (500 iters)
#
# Para ejecutar en background:
#   nohup bash 07_finetune_gpt_oss_20b.sh > logs/nohup-gpt-oss-20b.out 2>&1 &

set -e

PROJECT_DIR="/Users/jmfraga/Projects/medexpert-oncologia"
DATA_DIR="$PROJECT_DIR/data/clean"
ADAPTERS_DIR="$PROJECT_DIR/adapters"
MLX_ENV="/Users/jmfraga/mlx-env/bin"
MODEL="InferenceIllusionist/gpt-oss-20b-MLX-4bit"

# Detect pilot mode
if [ "$1" = "--pilot" ]; then
    echo "=== PILOT MODE (500 iters) ==="
    ITERS=500
    SAVE_EVERY=100
    STEPS_PER_EVAL=100
    ADAPTER_SUFFIX="gpt-oss-20b-onco-pilot"
else
    echo "=== FULL TRAINING (3000 iters) ==="
    ITERS=3000
    SAVE_EVERY=500
    STEPS_PER_EVAL=300
    ADAPTER_SUFFIX="gpt-oss-20b-onco"
fi

ADAPTER_PATH="$ADAPTERS_DIR/$ADAPTER_SUFFIX"
LOG_FILE="$PROJECT_DIR/logs/training-$ADAPTER_SUFFIX.log"
mkdir -p "$ADAPTER_PATH"
mkdir -p "$PROJECT_DIR/logs"

echo "Model: $MODEL"
echo "Data: $DATA_DIR"
echo "Adapters: $ADAPTER_PATH"
echo "Log: $LOG_FILE"
echo "Iterations: $ITERS"
echo "Batch size: 2 (x2 grad accum = effective 4)"
echo "Learning rate: 2e-5"
echo "Max seq length: 2048"
echo "LoRA rank: 64, scale: 32.0, dropout: 0.05"
echo ""

# Verify data exists
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "ERROR: $DATA_DIR/train.jsonl not found."
    exit 1
fi

TRAIN_COUNT=$(wc -l < "$DATA_DIR/train.jsonl")
VALID_COUNT=$(wc -l < "$DATA_DIR/valid.jsonl" 2>/dev/null || echo "0")
echo "Training examples: $TRAIN_COUNT"
echo "Validation examples: $VALID_COUNT"
echo ""

# Resume support
RESUME_FLAG=""
if [ -f "$ADAPTER_PATH/adapters.safetensors" ]; then
    echo "Found existing adapter, resuming..."
    RESUME_FLAG="--resume-adapter-file $ADAPTER_PATH"
fi

echo "Starting LoRA training at $(date)..."

$MLX_ENV/python -m mlx_lm lora \
    --model "$MODEL" \
    --train \
    --data "$DATA_DIR" \
    --adapter-path "$ADAPTER_PATH" \
    --batch-size 2 \
    --grad-accumulation-steps 2 \
    --num-layers -1 \
    --learning-rate 2e-5 \
    --iters "$ITERS" \
    --val-batches 50 \
    --steps-per-report 10 \
    --steps-per-eval "$STEPS_PER_EVAL" \
    --save-every "$SAVE_EVERY" \
    --max-seq-length 2048 \
    --grad-checkpoint \
    --mask-prompt \
    --config "$PROJECT_DIR/lora-config.yaml" \
    $RESUME_FLAG \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Training complete at $(date)"
echo "Adapters saved to: $ADAPTER_PATH"
echo ""

# Show final metrics
echo "=== Loss Summary ==="
grep "Val loss" "$LOG_FILE" | tail -10
