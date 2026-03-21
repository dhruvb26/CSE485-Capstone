#!/bin/bash
#SBATCH --job-name=vllm-eval
#SBATCH -p public
#SBATCH -q public
#SBATCH -A grp_ywang354
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --output=logs/jobs/slurm_%j.out
#SBATCH --error=logs/jobs/slurm_%j.err

set -euo pipefail

MODEL="${1:-Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled}"
CONFIG="${2:-}"
PORT=8000

export HF_HOME="/scratch/$USER/hf_models"
cd /home/$USER/projects/CSE485-Capstone
mkdir -p logs

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest
source activate /scratch/$USER/envs/venv

PYTHON=python

echo "Starting vLLM server for $MODEL on port $PORT ..."
$PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --dtype auto \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    &> logs/vllm_server_${SLURM_JOB_ID}.log &

VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

echo "Waiting for vLLM server to be ready ..."
MAX_WAIT=300
WAITED=0
until curl -sf http://localhost:${PORT}/health > /dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM server process died. Check logs/vllm_server_${SLURM_JOB_ID}.log"
        exit 1
    fi
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        echo "ERROR: vLLM server did not start within ${MAX_WAIT}s"
        kill "$VLLM_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "  ... waited ${WAITED}s"
done
echo "vLLM server ready after ${WAITED}s"

cleanup() {
    echo "Shutting down vLLM server (PID $VLLM_PID) ..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    echo "vLLM server stopped."
}
trap cleanup EXIT

export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

if [[ -z "$CONFIG" ]]; then
    $PYTHON -m eval
elif [[ "$CONFIG" == "score" ]]; then
    $PYTHON -m eval --evaluate-only
else
    $PYTHON -m eval --config "$CONFIG"
fi

echo "Job $SLURM_JOB_ID completed."
