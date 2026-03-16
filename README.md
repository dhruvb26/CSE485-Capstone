# ASU - AI for Business: Creating Smart Business Negotiation Bots

- [Overview](#overview)
- [Setup](#setup)
- [Training](#training)
- [Datasets](#datasets)
- [Troubleshooting](#troubleshooting)

## Overview

This project trains and evaluates LLMs as negotiation agents. Two LLMs are pitted against each other — one as a **buyer** and one as a **seller** — and negotiate over real product listings. The goal is to improve negotiation ability through supervised fine-tuning (SFT) and reinforcement learning (GRPO), then measure performance against a baseline model (GPT-4o).

**`benchmark/`** contains the initial implementations from our project to test out buyer/seller negotiation capabilities of the models. It runs multi-turn buyer-vs-seller dialogs between two LLMs and scores each negotiation on deal rate, profit, and action validity. A "testing model" is benchmarked in both roles (buyer and seller) against GPT-4o across hundreds of products.

**`rl/`** is the training pipeline. It uses GRPO (Group Relative Policy Optimization) and self-play to improve a model's negotiation strategy through reward-driven optimization. Rewards are based on deal success, price favorability, and format compliance. Built on [TRL](https://huggingface.co/docs/trl), [PEFT](https://huggingface.co/docs/peft), and [Transformers](https://huggingface.co/docs/transformers). Training metrics are visualized with [Trackio](https://github.com/gradio-app/trackio).

> **Helpful Resources:** [OS HuggingFace Cookbook](https://huggingface.co/learn/cookbook/index) and [HuggingFace LLM Course](https://huggingface.co/learn/llm-course/en/chapter0/1).

**`finetune/`** handles supervised fine-tuning with LoRA adapters. Models are first fine-tuned on negotiation dialog datasets before moving to RL training.

**`data/`** includes download scripts and preprocessing handlers for the negotiation datasets used across training and evaluation.

> **Need help?** Check out the [ASU Research Computing Guide](https://asurc.atlassian.net/wiki/spaces/RC/pages/2319417345/A+Brief+Example#Step-3---Use-/-Test) for detailed setup instructions.

## Setup

### 1. Get a GPU shell on SOL

```bash
interactive -p htc -t 2:00:00 --gres=gpu:a100:1
```

### 2. Load modules and create environment

```bash
module load mamba/latest
module load cuda-12.6.1-gcc-12.1.0

mamba env create -f environment.yml
conda activate venv
```

### 3. Verify GPU allocation (optional)

```bash
nvidia-smi
```

## Training

Submit training jobs to the SOL cluster using the `scripts/rl.sh` script:

```bash
sbatch scripts/rl.sh generate   # produce annotated SFT data via OpenAI-compatible API
sbatch scripts/rl.sh train      # run SFT training with LoRA
sbatch scripts/rl.sh grpo       # run GRPO training
sbatch scripts/rl.sh all        # run generate → train → grpo sequentially
```

Monitor job status and logs:

```bash
squeue -u $USER                 # check job status
cat logs/slurm_<job_id>.out     # view output
cat logs/slurm_<job_id>.err     # view errors
```

## Datasets

Datasets can be downloaded with the CLI tool:

```bash
python data/download.py --dataset <name>
```

Available datasets: `amazon_history_price`, `casino`, `craigslist_bargains`, `dnd`, `ji`, `ebay_best_offer`. See [`DATASETS.md`](data/DATASETS.md) for full details on each.

## Troubleshooting

### Common Issues

**Connection Problems:**

- Ensure Cisco VPN is active before SSH attempts
- Verify your ASU credentials are correct

**Environment Issues:**

- Verify CUDA module is loaded: `module list`
- Confirm virtual environment activation: `which python`

**Author identity unknown:**

`git config --global` wants to write to `~/.gitconfig` (which lives in $HOME). On certain SOL login or compute nodes, $HOME may be unset inside your job environment (it's a known bug with interactive sessions). So Git can't find the right location to store the global config.

To fix, run:

```bash
export HOME=/home/YOUR_ASU_ALIAS
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

**Course**: CSE 485/486 - Capstone Project I/II<br>
**Institution**: Arizona State University
