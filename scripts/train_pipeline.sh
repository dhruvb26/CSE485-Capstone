#!/bin/bash
#SBATCH --job-name=nego-train
#SBATCH -p general
#SBATCH -q private
#SBATCH -A grp_ywang354
#SBATCH -t 6:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=a100_80
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -euo pipefail

RUN_GENERATE=${RUN_GENERATE:-0}
RUN_SFT=${RUN_SFT:-0}
RUN_GRPO=${RUN_GRPO:-0}
SFT_DATA=data/sft/train_hindsight.jsonl

if [ "$RUN_GENERATE" = "0" ] && [ "$RUN_SFT" = "0" ] && [ "$RUN_GRPO" = "0" ]; then
    echo "No steps selected. Set at least one of: RUN_GENERATE=1 RUN_SFT=1 RUN_GRPO=1"
    exit 1
fi

export HF_HOME=/scratch/dbansa11/hf_models
cd /home/dbansa11/projects/CSE485-Capstone
module load cuda-12.6.1-gcc-12.1.0

PYTHON=/scratch/dbansa11/envs/venv/bin/python

mkdir -p logs

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Steps: generate=$RUN_GENERATE  sft=$RUN_SFT  grpo=$RUN_GRPO"
nvidia-smi -L
echo ""

if [ "$RUN_GENERATE" = "1" ]; then
    echo "Generating SFT data..."
    $PYTHON -m rl.sft.generate
    if [ ! -f "$SFT_DATA" ]; then
        echo "ERROR: SFT data not produced at $SFT_DATA"
        exit 1
    fi
    echo "Data generation complete: $SFT_DATA"
    echo ""
fi

if [ "$RUN_SFT" = "1" ]; then
    if [ ! -f "$SFT_DATA" ]; then
        echo "ERROR: SFT data not found at $SFT_DATA (run with RUN_GENERATE=1 first)"
        exit 1
    fi
    echo "Starting SFT training..."
    $PYTHON -m rl.sft.train
    echo "SFT complete. Adapter at checkpoints/sft-tuned/adapter"
    echo ""
fi

if [ "$RUN_GRPO" = "1" ]; then
    if [ ! -d "checkpoints/sft-tuned/adapter" ]; then
        echo "ERROR: SFT adapter not found (run with RUN_SFT=1 first)"
        exit 1
    fi
    echo "Starting GRPO..."
    $PYTHON -m rl.grpo.trainer
    echo "GRPO complete. Adapter at checkpoints/grpo/adapter"
    echo ""
fi
