#!/bin/bash
# Fase 4a — Fusionar adapters LoRA con modelo base
# Ejecutar en M4
#
# Uso:
#   bash 04_fuse_model.sh              # Fuse full adapters
#   bash 04_fuse_model.sh --pilot      # Fuse pilot adapters

set -e

PROJECT_DIR="/Users/jmfraga/Projects/medexpert-oncologia"
MLX_ENV="/Users/jmfraga/mlx-env/bin"
MODEL="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
OUTPUT_DIR="$PROJECT_DIR/models/Llama8B-MedExpert-Oncologia"

if [ "$1" = "--pilot" ]; then
    ADAPTER_PATH="$PROJECT_DIR/adapters/lora-pilot"
    OUTPUT_DIR="${OUTPUT_DIR}-pilot"
else
    ADAPTER_PATH="$PROJECT_DIR/adapters/lora-full"
fi

echo "Fusing model..."
echo "  Base: $MODEL"
echo "  Adapters: $ADAPTER_PATH"
echo "  Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

$MLX_ENV/python -m mlx_lm.fuse \
    --model "$MODEL" \
    --adapter-path "$ADAPTER_PATH" \
    --save-path "$OUTPUT_DIR" \
    --de-quantize

echo ""
echo "Model fused to: $OUTPUT_DIR"
echo "Size: $(du -sh $OUTPUT_DIR | cut -f1)"
