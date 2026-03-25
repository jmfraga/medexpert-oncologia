#!/bin/bash
# Fase 3 — Fine-tuning con MLX LoRA
# Ejecutar en M4 (64GB RAM, Apple Silicon)
#
# Uso:
#   bash 03_finetune_mlx.sh              # Run completo
#   bash 03_finetune_mlx.sh --pilot      # Run piloto (500 iters)

set -e

PROJECT_DIR="/Users/jmfraga/Projects/medexpert-oncologia"
DATA_DIR="$PROJECT_DIR/data"
ADAPTERS_DIR="$PROJECT_DIR/adapters"
MLX_ENV="/Users/jmfraga/mlx-env/bin"
MODEL="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"

# Detect pilot mode
if [ "$1" = "--pilot" ]; then
    echo "=== PILOT MODE (500 iters) ==="
    ITERS=500
    SAVE_EVERY=100
    STEPS_PER_EVAL=100
    ADAPTER_SUFFIX="pilot"
else
    echo "=== FULL TRAINING ==="
    ITERS=3000
    SAVE_EVERY=500
    STEPS_PER_EVAL=200
    ADAPTER_SUFFIX="full"
fi

ADAPTER_PATH="$ADAPTERS_DIR/lora-$ADAPTER_SUFFIX"
mkdir -p "$ADAPTER_PATH"

echo "Model: $MODEL"
echo "Data: $DATA_DIR"
echo "Adapters: $ADAPTER_PATH"
echo "Iterations: $ITERS"
echo ""

# Verify data exists
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "ERROR: $DATA_DIR/train.jsonl not found. Run 02_generate_dataset.py first."
    exit 1
fi

TRAIN_COUNT=$(wc -l < "$DATA_DIR/train.jsonl")
VALID_COUNT=$(wc -l < "$DATA_DIR/valid.jsonl" 2>/dev/null || echo "0")
echo "Training examples: $TRAIN_COUNT"
echo "Validation examples: $VALID_COUNT"
echo ""

# ── Run LoRA training ──
echo "Starting LoRA training at $(date)..."
echo "Logs: $PROJECT_DIR/logs/training-$ADAPTER_SUFFIX.log"

$MLX_ENV/python -m mlx_lm.lora \
    --model "$MODEL" \
    --train \
    --data "$DATA_DIR" \
    --adapter-path "$ADAPTER_PATH" \
    --batch-size 4 \
    --lora-layers -1 \
    --learning-rate 2e-5 \
    --iters "$ITERS" \
    --val-batches 50 \
    --steps-per-report 10 \
    --steps-per-eval "$STEPS_PER_EVAL" \
    --save-every "$SAVE_EVERY" \
    --max-seq-length 2048 \
    --grad-checkpoint \
    --mask-prompt \
    --lora-parameters '{"rank": 64, "dropout": 0.05, "scale": 32.0}' \
    2>&1 | tee "$PROJECT_DIR/logs/training-$ADAPTER_SUFFIX.log"

echo ""
echo "Training complete at $(date)"
echo "Adapters saved to: $ADAPTER_PATH"
echo ""

# Show final metrics
echo "=== Loss Summary ==="
grep "Val loss" "$PROJECT_DIR/logs/training-$ADAPTER_SUFFIX.log" | tail -5
