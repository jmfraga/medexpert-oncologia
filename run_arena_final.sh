#!/bin/bash
# Arena Final — Automated orchestrator
# Runs from M4, coordinates MLX servers locally and arena runner on M1
#
# Sequence:
# 1. For each model: start MLX servers (FT + base) → trigger runner on M1 → stop servers
# 2. Run Sonnet tier (API only, no local server needed)
# 3. Run Opus judge on M1
#
# Usage: bash run_arena_final.sh

set -e

MLX_ENV="/Users/jmfraga/mlx-env/bin"
PROJECT_DIR="/Users/jmfraga/Projects/medexpert-oncologia"
M1_SSH="juanma@100.107.30.22"
M1_VENV="/Users/juanma/Projects/medexpert-admin/venv/bin/python"
M1_WORKDIR="/Users/juanma/Projects/medexpert-admin"
RESULTS_FILE=""  # Will be set after first run

export HF_HUB_OFFLINE=1

wait_for_server() {
    local port=$1
    local max_wait=$2
    local elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; then
            echo "    Port $port: ready"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "    Port $port: TIMEOUT after ${max_wait}s"
    return 1
}

run_tiers_on_m1() {
    local tiers="$1"
    local merge_flag=""
    if [ -n "$RESULTS_FILE" ]; then
        merge_flag="--merge $RESULTS_FILE"
    fi

    echo "  Running on M1: tiers=$tiers"
    # Run arena runner on M1 and capture output path
    local output
    output=$(ssh $M1_SSH "cd $M1_WORKDIR && $M1_VENV arena/arena_runner.py --tiers $tiers $merge_flag 2>&1" | tee /dev/stderr | grep "Resultados guardados" | awk '{print $NF}')

    if [ -n "$output" ]; then
        RESULTS_FILE="$output"
        echo "  Results saved to: $RESULTS_FILE"
    else
        echo "  WARNING: Could not capture output path"
        # Find latest results file
        RESULTS_FILE=$(ssh $M1_SSH "ls -t $M1_WORKDIR/arena/results/responses_*.json | head -1")
        echo "  Using latest: $RESULTS_FILE"
    fi
}

serve_and_run() {
    local MODEL="$1"
    local ADAPTER="$2"
    local PORT_FT="$3"
    local PORT_BASE="$4"
    local TIER_FT="$5"
    local TIER_BASE="$6"
    local NAME="$7"

    echo ""
    echo "========================================"
    echo "  $NAME"
    echo "========================================"

    # Start FT server
    echo "  Starting FT server (port $PORT_FT)..."
    $MLX_ENV/python -m mlx_lm.server \
        --model "$MODEL" \
        --adapter-path "$ADAPTER" \
        --host 0.0.0.0 --port "$PORT_FT" > /tmp/mlx_arena_ft.log 2>&1 &
    PID_FT=$!

    # Start base server
    echo "  Starting base server (port $PORT_BASE)..."
    $MLX_ENV/python -m mlx_lm.server \
        --model "$MODEL" \
        --host 0.0.0.0 --port "$PORT_BASE" > /tmp/mlx_arena_base.log 2>&1 &
    PID_BASE=$!

    # Wait for both servers
    echo "  Waiting for servers..."
    wait_for_server $PORT_FT 120
    wait_for_server $PORT_BASE 120

    # Run the 2 tiers on M1
    run_tiers_on_m1 "$TIER_FT $TIER_BASE"

    # Stop servers
    echo "  Stopping servers..."
    kill $PID_FT $PID_BASE 2>/dev/null || true
    wait $PID_FT $PID_BASE 2>/dev/null || true
    sleep 5
    echo "  Servers stopped."
}

echo "=========================================="
echo "  MedExpert Arena Final"
echo "  7 tiers × 15 casos = 105 evaluaciones"
echo "=========================================="
echo ""
echo "Start time: $(date)"

# ── Round 1: Gemma4 31B ──
serve_and_run \
    "mlx-community/gemma-4-31b-it-4bit" \
    "$PROJECT_DIR/adapters/gemma4-31b-onco" \
    8090 8093 \
    "ft_gemma4_31b" "base_gemma4_31b_rag" \
    "Gemma4 31B (FT + base+RAG)"

# ── Round 2: GPT-oss 20B ──
serve_and_run \
    "InferenceIllusionist/gpt-oss-20b-MLX-4bit" \
    "$PROJECT_DIR/adapters/gpt-oss-20b-onco" \
    8091 8094 \
    "ft_gptoss_20b" "base_gptoss_20b_rag" \
    "GPT-oss 20B (FT + base+RAG)"

# ── Round 3: Gemma4 26B MoE ──
serve_and_run \
    "mlx-community/gemma-4-26b-a4b-it-4bit" \
    "$PROJECT_DIR/adapters/gemma4-26b-moe-onco" \
    8092 8095 \
    "ft_gemma4_26b_moe" "base_gemma4_26b_moe_rag" \
    "Gemma4 26B MoE (FT + base+RAG)"

# ── Round 4: Sonnet (API only) ──
echo ""
echo "========================================"
echo "  Sonnet 4.6 (API)"
echo "========================================"
run_tiers_on_m1 "sonnet_norag"

echo ""
echo "========================================"
echo "  All 7 tiers complete!"
echo "  Results: $RESULTS_FILE"
echo "========================================"

# ── Round 5: Opus Judge ──
echo ""
echo "Running Opus judge on all 105 responses..."
ssh $M1_SSH "cd $M1_WORKDIR && $M1_VENV arena/arena_judge.py 2>&1" | tee /tmp/arena_judge.log

echo ""
echo "========================================"
echo "  Arena Final COMPLETE"
echo "  End time: $(date)"
echo "========================================"
