#!/bin/bash
#SBATCH --job-name=eval
#SBATCH -p public
#SBATCH -q public
#SBATCH -A grp_ywang354
#SBATCH -t 3:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --output=logs/jobs/slurm_%j.out
#SBATCH --error=logs/jobs/slurm_%j.err

set -euo pipefail

# --- Usage ----------------------------------------------------------------
# sbatch scripts/eval.sh tasks                      # run tasks (default config)
# sbatch scripts/eval.sh tasks path/to/config.yaml  # run tasks (custom config)
# sbatch scripts/eval.sh tasks score                # score existing task logs
# sbatch scripts/eval.sh tasks list                 # list available tasks
#
# sbatch scripts/eval.sh negotiate                  # run negotiate (default config)
# sbatch scripts/eval.sh negotiate path/to/cfg.yaml # run negotiate (custom config)
# sbatch scripts/eval.sh negotiate score logs/negotiate/run_...  # score existing logs
#
# sbatch scripts/eval.sh vllm-tasks [MODEL] [CONFIG] # start vLLM server + run tasks
# sbatch scripts/eval.sh vllm-negotiate MODEL [CONFIG] # start vLLM + run negotiate
# --------------------------------------------------------------------------

MODE="${1:-tasks}"
shift || true

export HF_HOME="/scratch/$USER/hf_models"
cd /home/$USER/projects/CSE485-Capstone
mkdir -p logs/jobs

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest
source activate /scratch/$USER/envs/venv

PYTHON=python

start_vllm() {
    local MODEL="$1"
    local PORT="${2:-8000}"

    echo "Starting vLLM server for $MODEL on port $PORT ..."
    $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --model-impl transformers \
        --port "$PORT" \
        --dtype auto \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.90 \
        &> "logs/jobs/vllm_server_${SLURM_JOB_ID}.log" &

    VLLM_PID=$!
    echo "vLLM server PID: $VLLM_PID"

    cleanup() {
        echo "Shutting down vLLM server (PID $VLLM_PID) ..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        echo "vLLM server stopped."
    }
    trap cleanup EXIT

    export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

    echo "Waiting for vLLM server to be ready ..."
    for i in $(seq 1 120); do
        if curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
            echo "vLLM server ready."
            return 0
        fi
        sleep 5
    done
    echo "ERROR: vLLM server failed to start within 10 minutes."
    exit 1
}

case "$MODE" in
    tasks)
        ARG="${1:-}"
        if [[ -z "$ARG" ]]; then
            $PYTHON -m eval tasks
        elif [[ "$ARG" == "score" ]]; then
            $PYTHON -m eval tasks --evaluate-only
        elif [[ "$ARG" == "list" ]]; then
            $PYTHON -m eval tasks --list-tasks
        else
            $PYTHON -m eval tasks --config "$ARG"
        fi
        ;;

    negotiate)
        ARG="${1:-}"
        if [[ -z "$ARG" ]]; then
            $PYTHON -m eval negotiate
        elif [[ "$ARG" == "score" ]]; then
            LOG_DIR="${2:?negotiate score requires a LOG_DIR argument}"
            $PYTHON -m eval negotiate --evaluate-only "$LOG_DIR"
        else
            $PYTHON -m eval negotiate --config "$ARG"
        fi
        ;;

    vllm-tasks)
        MODEL="${1:-Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled}"
        CONFIG="${2:-}"

        start_vllm "$MODEL"

        if [[ -z "$CONFIG" ]]; then
            $PYTHON -m eval tasks
        elif [[ "$CONFIG" == "score" ]]; then
            $PYTHON -m eval tasks --evaluate-only
        else
            $PYTHON -m eval tasks --config "$CONFIG"
        fi
        ;;

    vllm-negotiate)
        MODEL="${1:?vllm-negotiate requires a MODEL argument}"
        CONFIG="${2:-}"

        start_vllm "$MODEL"

        if [[ -z "$CONFIG" ]]; then
            $PYTHON -m eval negotiate
        else
            $PYTHON -m eval negotiate --config "$CONFIG"
        fi
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 {tasks|negotiate|vllm-tasks|vllm-negotiate} [args...]"
        exit 1
        ;;
esac

echo "Job $SLURM_JOB_ID completed."
