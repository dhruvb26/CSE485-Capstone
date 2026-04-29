#!/bin/bash
#SBATCH --job-name=eval
#SBATCH -p public
#SBATCH -q public
#SBATCH -A grp_ywang354
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH --output=logs/jobs/eval_%j.out
#SBATCH --error=logs/jobs/eval_%j.err

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
# sbatch scripts/eval.sh vllm-tasks [CONFIG]      # auto-detect local model, start vLLM + tasks
# sbatch scripts/eval.sh vllm-negotiate [CONFIG]  # auto-detect local model, start vLLM + negotiate
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

PYTHON=/scratch/$USER/envs/venv/bin/python

detect_local_model() {
    local CONFIG="$1"
    $PYTHON -c "
import yaml, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
for m in cfg.get('matchups', []):
    for role in ('learner', 'opponent'):
        agent = m.get(role, {})
        url = agent.get('base_url', '')
        if 'localhost' in url or '127.0.0.1' in url:
            print(agent['model'])
            sys.exit(0)
for m in cfg.get('models', []):
    if m.get('type') == 'vllm_model':
        print(m.get('model_str', ''))
        sys.exit(0)
print('', file=sys.stderr)
sys.exit(1)
"
}

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
        CONFIG="${1:-$(dirname "$0")/../eval/configs/tasks.yaml}"

        MODEL=$(detect_local_model "$CONFIG") || {
            echo "ERROR: no localhost model found in $CONFIG"; exit 1
        }
        start_vllm "$MODEL"

        $PYTHON -m eval tasks --config "$CONFIG"
        ;;

    vllm-negotiate)
        CONFIG="${1:-$(dirname "$0")/../eval/configs/negotiate.yaml}"

        MODEL=$(detect_local_model "$CONFIG") || {
            echo "ERROR: no localhost model found in $CONFIG"; exit 1
        }
        start_vllm "$MODEL"

        $PYTHON -m eval negotiate --config "$CONFIG"
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 {tasks|negotiate|vllm-tasks|vllm-negotiate} [args...]"
        exit 1
        ;;
esac

echo "Job $SLURM_JOB_ID completed."
