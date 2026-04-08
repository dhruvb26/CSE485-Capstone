# ASU - AI for Business: Creating Smart Business Negotiation Bots

- [Overview](#overview)
- [Setup](#setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Datasets](#datasets)
- [Charts](#charts)
- [Report](#report)
- [Troubleshooting](#troubleshooting)

## Overview

Check out [Google Drive](https://drive.google.com/drive/folders/10iKqBuJy0eDGJXZf4kveI-YooZ7WTY5n?usp=sharing) for all documents, reports and product reports.

**`benchmark/`** contains the initial implementations from our project to test out buyer/seller negotiation capabilities of the models. It runs multi-turn buyer-vs-seller dialogs between two LLMs and scores each negotiation on deal rate, profit, and action validity. A "testing model" is benchmarked in both roles (buyer and seller) against GPT-4o across hundreds of products.

> **Note:** This module should be built upon in the future, when we have decided on the final architecture to be used for our negotiation bot. For now, it remains as a placeholder for future use with some legacy code.

**`rl/`** is the training pipeline. It uses GRPO (Group Relative Policy Optimization) and self-play to improve a model's negotiation strategy through reward-driven optimization. Rewards are based on deal success, price favorability, and format compliance. Built on [TRL](https://huggingface.co/docs/trl), [PEFT](https://huggingface.co/docs/peft), and [Transformers](https://huggingface.co/docs/transformers). Training metrics are visualized with [Trackio](https://github.com/gradio-app/trackio).

> **Helpful Resources:** [OS HuggingFace Cookbook](https://huggingface.co/learn/cookbook/index) and [HuggingFace LLM Course](https://huggingface.co/learn/llm-course/en/chapter0/1).

**`finetune/`** handles supervised fine-tuning with LoRA adapters. Models are first fine-tuned on negotiation dialog datasets before moving to RL training.

**`eval/`** is the evaluation framework, adapted from [SysEval-NegoLLMs](https://github.com/DSincerity/SysEval-NegoLLMs). It systematically assesses LLM negotiation capabilities across 4 datasets (Deal or No Deal, CaSiNo, Job Interview, CRA) and 3 dialogue stages (start, mid, end). Supports OpenAI models, HuggingFace models, and custom-trained local checkpoints (including LoRA adapters). Configured via a single YAML file. See [`eval/README.md`](eval/README.md) for full documentation.

**`data/`** includes download scripts and preprocessing handlers for the negotiation datasets used across training and evaluation.

> **Need help?** Check out the [ASU Research Computing Guide](https://asurc.atlassian.net/wiki/spaces/RC/pages/2319417345/A+Brief+Example#Step-3---Use-/-Test) for detailed setup instructions.

## Setup

Training and evaluation run on [ASU Research Computing (SOL)](https://docs.rc.asu.edu/), which provides A100 GPUs via SLURM. For lightweight tasks like generating charts, compiling the report, or running eval scripts locally, we use [uv](https://docs.astral.sh/uv/) as the Python package manager.

### SOL (GPU training & evaluation)

#### 1. Get a GPU shell on SOL

```bash
interactive -p htc -t 2:00:00 --gres=gpu:a100:1
```

#### 2. Load modules and create environment

```bash
module load mamba/latest
module load cuda-12.6.1-gcc-12.1.0

mamba env create -f environment.yml
conda activate venv
```

#### 3. Verify GPU allocation (optional)

```bash
nvidia-smi
```

### Local (charts, eval, report)

#### 1. Install uv

```bash
brew install uv
```

#### 2. Sync dependencies

```bash
uv sync            # core dependencies
uv sync --extra eval  # include eval dependencies
```

This reads `pyproject.toml` and creates a `.venv` automatically.

#### 3. Run scripts

```bash
uv run scripts/download_charts.py
uv run -m eval --list-tasks
```

No manual `venv` activation needed — `uv run` handles it.

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

## Evaluation

Run systematic evaluations of any model on negotiation tasks. Configure `eval/config.yaml` with your models and tasks, then:

```bash
python -m eval                              # run with default config
python -m eval --config path/to/config.yaml # custom config
python -m eval --evaluate-only              # score existing logs only
python -m eval --list-tasks                 # list all available tasks
```

Example config for evaluating a local Qwen model on all CaSiNo tasks:

```yaml
models:
  - type: local_model
    model_path: Qwen/Qwen2.5-3B-Instruct
    label: qwen2.5-3b-instruct

tasks:
  - all_casino
```

To compare your fine-tuned checkpoint against the base model:

```yaml
models:
  - type: local_model
    model_path: Qwen/Qwen2.5-3B-Instruct
    label: qwen2.5-3b-base

  - type: local_model
    model_path: checkpoints/grpo-tuned
    base_model: Qwen/Qwen2.5-3B-Instruct
    label: qwen2.5-3b-grpo

tasks:
  - all
```

Results are saved as JSON under `logs/eval/` (gitignored). See [`README.md`](eval/README.md) for all available tasks, metrics, and config options.

## Datasets

Datasets can be downloaded with the CLI tool:

```bash
python data/download.py --dataset <name>
```

Available datasets: `amazon_history_price`, `casino`, `cra`, `craigslist_bargains`, `dnd`, `ji`, `ebay_best_offer`. See [`DATASETS.md`](data/DATASETS.md) for full details on each.

## Charts

The `scripts/download_charts.py` script fetches training metrics from the [Trackio](https://github.com/gradio-app/trackio) HuggingFace Space and generates chart PNGs into `assets/`. Dependencies (`trackio`, `matplotlib`, `huggingface-hub`) are handled by `uv sync` (see [Local setup](#local-charts-eval-report) above).

### HuggingFace Authentication

The Trackio CLI needs access to the HF Space. Either log in interactively:

```bash
huggingface-cli login
```

Or set the token as an environment variable:

```bash
export HF_TOKEN=hf_...
```

### Usage

```bash
uv run scripts/download_charts.py              # all run groups, dark theme
uv run scripts/download_charts.py --light       # light theme
uv run scripts/download_charts.py --run sft-0328-0839  # single run group
```

Charts are saved to `assets/<group-name>/` (e.g. `assets/sft/`, `assets/grpo-selfplay/`).

## Report

The project white paper lives in `report/` and is compiled with [Pandoc](https://pandoc.org/).

### Install Pandoc (macOS)

```bash
brew install pandoc
```

You also need a LaTeX distribution for PDF output:

```bash
brew install --cask mactex       # full install (~4 GB)
# or
brew install --cask basictex     # minimal install (~100 MB)
```

If using BasicTeX, you may need to install extra packages:

```bash
sudo tlmgr update --self
sudo tlmgr install booktabs float collection-fontsrecommended
```

### Compile

```bash
cd report
pandoc report.md -o report.pdf --citeproc
```

The YAML frontmatter in `report.md` handles all configuration (bibliography, citation style, layout). The `--citeproc` flag processes citations from `references.bib` using the IEEE style defined in `ieee.csl`.

## Troubleshooting

### Common Issues

**Connection Problems:**

- Ensure Cisco VPN is active before SSH attempts
- Verify your ASU credentials are correct

**Environment Issues:**

- Verify CUDA module is loaded: `module list`
- Confirm virtual environment activation: `which python`

**Running out of disk space (model downloads):**

By default, HuggingFace downloads models to `~/.cache/huggingface/`, which sits on the limited home directory. Redirect it to scratch storage:

```bash
export HF_HOME=/scratch/$USER/hf_models
```

Add this to your `~/.bashrc` or job script so it persists across sessions.

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
