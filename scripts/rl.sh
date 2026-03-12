#!/bin/bash
#SBATCH --job-name=rl-train
#SBATCH -p general
#SBATCH -q private
#SBATCH -A grp_ywang354
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -euo pipefail

TASK="${1:-}"

if [[ -z "$TASK" ]] || [[ "$TASK" != "generate" && "$TASK" != "train" && "$TASK" != "grpo" && "$TASK" != "all" ]]; then
    echo "Usage: sbatch $0 {generate|train|grpo|all}"
    echo "  generate  — run GPT annotation to produce SFT data"
    echo "  train     — run SFT training with LoRA"
    echo "  grpo      — run GRPO training with self-play rollouts"
    echo "  all       — run generate then train"
    exit 1
fi

export HF_HOME="/scratch/dbansa11/hf_models"
cd /home/dbansa11/projects/CSE485-Capstone
mkdir -p logs

module load cuda-12.6.1-gcc-12.1.0

PYTHON=/scratch/dbansa11/envs/venv/bin/python
GENERATE_CFG="rl/configs/generate.yaml"
VLLM_PID=""

start_vllm() {
    local mode model port
    mode=$($PYTHON -c "import yaml; print(yaml.safe_load(open('$GENERATE_CFG'))['mode'])")
    [[ "$mode" != "local" ]] && return 0

    model=$($PYTHON -c "import yaml; print(yaml.safe_load(open('$GENERATE_CFG'))['local']['model'])")
    port=8000

    local num_gpus
    num_gpus=$(nvidia-smi -L | wc -l)

    echo "Starting vLLM server: model=$model port=$port tp=$num_gpus"
    $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --port "$port" \
        --tensor-parallel-size "$num_gpus" \
        --max-model-len 8192 &
    VLLM_PID=$!

    echo "Waiting for vLLM server (pid=$VLLM_PID) to be ready..."
    local elapsed=0
    until curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM server died during startup"
            exit 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "vLLM server ready after ${elapsed}s"
}

stop_vllm() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM server (pid=$VLLM_PID)"
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
}

trap stop_vllm EXIT

run_generate() {
    start_vllm
    $PYTHON -m rl.main generate
    stop_vllm
}

run_train() {
    $PYTHON -m rl.main train
}

run_grpo() {
    $PYTHON -m rl.main grpo
}

if [[ "$TASK" == "all" ]]; then
    run_generate
    run_train
elif [[ "$TASK" == "generate" ]]; then
    run_generate
elif [[ "$TASK" == "grpo" ]]; then
    run_grpo
else
    run_train
fi

echo "Job $SLURM_JOB_ID completed."
