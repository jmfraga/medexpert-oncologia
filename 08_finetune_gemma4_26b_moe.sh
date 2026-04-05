#!/bin/bash
# Fine-tune Gemma 4 26B MoE (A4B, 4-bit) for MedExpert Oncology
# Ejecutar en M4 (64GB RAM, Apple Silicon)
# Requiere mlx-lm >= 0.31.2 con soporte MoE (PR #1093)
#
# Uso:
#   bash 08_finetune_gemma4_26b_moe.sh              # Run completo (3000 iters)
#   bash 08_finetune_gemma4_26b_moe.sh --pilot       # Run piloto (500 iters)
#
# Para ejecutar en background:
#   nohup bash 08_finetune_gemma4_26b_moe.sh > /dev/null 2>&1 &

set -e

# Use cached model (avoid re-downloading updated revision)
export HF_HUB_OFFLINE=1

PROJECT_DIR="/Users/jmfraga/Projects/medexpert-oncologia"
DATA_DIR="$PROJECT_DIR/data/clean"
ADAPTERS_DIR="$PROJECT_DIR/adapters"
MLX_ENV="/Users/jmfraga/mlx-env/bin"
MODEL="mlx-community/gemma-4-26b-a4b-it-4bit"

# Detect pilot mode
if [ "$1" = "--pilot" ]; then
    echo "=== PILOT MODE (500 iters) ==="
    ITERS=500
    SAVE_EVERY=100
    STEPS_PER_EVAL=100
    ADAPTER_SUFFIX="gemma4-26b-moe-onco-pilot"
else
    echo "=== FULL TRAINING (3000 iters) ==="
    ITERS=3000
    SAVE_EVERY=500
    STEPS_PER_EVAL=300
    ADAPTER_SUFFIX="gemma4-26b-moe-onco"
fi

ADAPTER_PATH="$ADAPTERS_DIR/$ADAPTER_SUFFIX"
LOG_FILE="$PROJECT_DIR/logs/training-$ADAPTER_SUFFIX.log"
mkdir -p "$ADAPTER_PATH"

echo "Model: $MODEL"
echo "Data: $DATA_DIR"
echo "Adapters: $ADAPTER_PATH"
echo "Log: $LOG_FILE"
echo "Iterations: $ITERS"
echo "Batch size: 1 (x4 grad accum = effective 4)"
echo "Learning rate: 2e-5"
echo "Max seq length: 2048"
echo "LoRA rank: 8, scale: 16.0, dropout: 0.05 (low rank for 128 experts)"
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
    --batch-size 1 \
    --grad-accumulation-steps 4 \
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
    --config "$PROJECT_DIR/lora-config-gemma4-26b-moe.yaml" \
    $RESUME_FLAG \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Training complete at $(date)"
echo "Adapters saved to: $ADAPTER_PATH"
echo ""

# Show final metrics
echo "=== Loss Summary ==="
grep "Val loss" "$LOG_FILE" | tail -10
