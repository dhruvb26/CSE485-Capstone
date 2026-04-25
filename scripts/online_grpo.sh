#!/bin/bash
#SBATCH --job-name=rlvr-train
#SBATCH -p general
#SBATCH -q grp_ywang354
#SBATCH -A grp_ywang354
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100_80
#SBATCH --mem-per-gpu=64G
#SBATCH -c 24
#SBATCH --output=logs/jobs/slurm_%j.out
#SBATCH --error=logs/jobs/slurm_%j.err

set -euo pipefail

CONFIG="${1:-rl/configs/online_grpo.yaml}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Config file not found: $CONFIG"
    echo "Usage: sbatch $0 [path/to/config.yaml]"
    exit 1
fi

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

echo "Starting online GRPO training with config: $CONFIG"
echo "GPUs visible: ${CUDA_VISIBLE_DEVICES:-all}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

$PYTHON -m rl.online_grpo --config "$CONFIG"

echo "Job $SLURM_JOB_ID completed."
