#!/bin/bash
#SBATCH --job-name=eval
#SBATCH -p htc
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

CONFIG="${1:-}"

export HF_HOME="/scratch/$USER/hf_models"
cd /home/$USER/projects/CSE485-Capstone
mkdir -p logs

module load cuda-12.6.1-gcc-12.1.0

PYTHON=/scratch/$USER/envs/venv/bin/python

if [[ -z "$CONFIG" ]]; then
    $PYTHON -m eval
elif [[ "$CONFIG" == "score" ]]; then
    $PYTHON -m eval --evaluate-only
elif [[ "$CONFIG" == "list" ]]; then
    $PYTHON -m eval --list-tasks
else
    $PYTHON -m eval --config "$CONFIG"
fi

echo "Job $SLURM_JOB_ID completed."
