#!/bin/bash
# Arena Final — MLX server orchestrator
# Levanta modelos secuencialmente para la Arena Final
# Cada modelo se sirve en 2 puertos: FT (con adapter) y base (sin adapter)
#
# Uso: bash arena_serve.sh <model_group>
#   model_group: gemma4_31b | gptoss_20b | gemma4_26b_moe | all
#
# "all" corre los 3 grupos secuencialmente, esperando señal entre cada uno.

set -e

MLX_ENV="/Users/jmfraga/mlx-env/bin"
PROJECT_DIR="/Users/jmfraga/Projects/medexpert-oncologia"
export HF_HUB_OFFLINE=1

serve_model() {
    local MODEL="$1"
    local ADAPTER="$2"
    local PORT_FT="$3"
    local PORT_BASE="$4"
    local NAME="$5"

    echo ""
    echo "============================================"
    echo "  Serving: $NAME"
    echo "  Model: $MODEL"
    echo "  FT port: $PORT_FT (with adapter)"
    echo "  Base port: $PORT_BASE (without adapter)"
    echo "============================================"

    # Start FT server (with adapter)
    echo "[$(date)] Starting FT server on port $PORT_FT..."
    $MLX_ENV/python -m mlx_lm.server \
        --model "$MODEL" \
        --adapter-path "$ADAPTER" \
        --port "$PORT_FT" &
    PID_FT=$!

    # Start base server (without adapter)
    echo "[$(date)] Starting base server on port $PORT_BASE..."
    $MLX_ENV/python -m mlx_lm.server \
        --model "$MODEL" \
        --port "$PORT_BASE" &
    PID_BASE=$!

    # Wait for servers to be ready
    echo "Waiting for servers to start..."
    sleep 15

    # Health check
    for port in $PORT_FT $PORT_BASE; do
        if curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; then
            echo "  Port $port: OK"
        else
            echo "  Port $port: waiting more..."
            sleep 15
            if curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; then
                echo "  Port $port: OK"
            else
                echo "  Port $port: FAILED"
            fi
        fi
    done

    echo ""
    echo ">>> Servers ready for $NAME"
    echo ">>> Run the arena tiers now, then press ENTER to stop and continue."
    read -r

    # Kill servers
    echo "[$(date)] Stopping servers..."
    kill $PID_FT $PID_BASE 2>/dev/null
    wait $PID_FT $PID_BASE 2>/dev/null
    sleep 5
    echo "Servers stopped."
}

case "${1:-all}" in
    gemma4_31b)
        serve_model \
            "mlx-community/gemma-4-31b-it-4bit" \
            "$PROJECT_DIR/adapters/gemma4-31b-onco" \
            8090 8093 "Gemma4 31B"
        ;;
    gptoss_20b)
        serve_model \
            "InferenceIllusionist/gpt-oss-20b-MLX-4bit" \
            "$PROJECT_DIR/adapters/gpt-oss-20b-onco" \
            8091 8094 "GPT-oss 20B"
        ;;
    gemma4_26b_moe)
        serve_model \
            "mlx-community/gemma-4-26b-a4b-it-4bit" \
            "$PROJECT_DIR/adapters/gemma4-26b-moe-onco" \
            8092 8095 "Gemma4 26B MoE"
        ;;
    all)
        echo "=== Arena Final — Sequential Model Serving ==="
        echo "Will serve 3 models one at a time."
        echo ""

        serve_model \
            "mlx-community/gemma-4-31b-it-4bit" \
            "$PROJECT_DIR/adapters/gemma4-31b-onco" \
            8090 8093 "Gemma4 31B"

        serve_model \
            "InferenceIllusionist/gpt-oss-20b-MLX-4bit" \
            "$PROJECT_DIR/adapters/gpt-oss-20b-onco" \
            8091 8094 "GPT-oss 20B"

        serve_model \
            "mlx-community/gemma-4-26b-a4b-it-4bit" \
            "$PROJECT_DIR/adapters/gemma4-26b-moe-onco" \
            8092 8095 "Gemma4 26B MoE"

        echo ""
        echo "=== All models served. Arena complete! ==="
        ;;
    *)
        echo "Usage: $0 {gemma4_31b|gptoss_20b|gemma4_26b_moe|all}"
        exit 1
        ;;
esac
