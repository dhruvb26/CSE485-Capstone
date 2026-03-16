#!/bin/bash
#SBATCH --job-name=rl-train
#SBATCH -p htc
#SBATCH -q public
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
    echo "  generate  — call OpenAI-compatible API to produce annotated SFT data"
    echo "  train     — run SFT training with LoRA"
    echo "  grpo      — run GRPO training (annotated or self-play, set in grpo.yaml)"
    echo "  all       — run generate, then train, then grpo"
    exit 1
fi

export HF_HOME="/scratch/dbansa11/hf_models"
cd /home/dbansa11/projects/CSE485-Capstone
mkdir -p logs

module load cuda-12.6.1-gcc-12.1.0

PYTHON=/scratch/dbansa11/envs/venv/bin/python

run_generate() {
    $PYTHON -m rl.main generate
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
    run_grpo
elif [[ "$TASK" == "generate" ]]; then
    run_generate
elif [[ "$TASK" == "grpo" ]]; then
    run_grpo
else
    run_train
fi

echo "Job $SLURM_JOB_ID completed."
