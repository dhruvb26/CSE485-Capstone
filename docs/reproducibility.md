# Reproducibility Checklist

This checklist keeps the final CaSiNo/GRPO project runnable from a fresh clone
and makes the handoff clearer for future students.

## Canonical Project Version

Use `main` as the source of truth for the final capstone submission. The final
pipeline is:

1. Generate annotated CaSiNo conversations with `rl.main generate`.
2. Train the LoRA SFT checkpoint with `rl.main sft`.
3. Run annotated or self-play GRPO with `rl.main grpo`.
4. Evaluate with the task framework in `eval/` and the head-to-head harness in
   `eval/negotiate.py`.

The older Amazon price-history buyer/seller benchmark remains in `benchmark/`
as legacy context. It is not the final report pipeline.

## Fresh Clone Checks

From a clean checkout:

```bash
git status --short --branch
uv sync
uv sync --extra eval
uv run -m eval --list-tasks
```

Expected result: the repo is clean and the eval task list prints without import
errors.

## SOL Setup

On SOL, create log directories before submitting SLURM jobs because the scripts
write job output under `logs/jobs/`:

```bash
mkdir -p logs/jobs
module load mamba/latest
module load cuda-12.6.1-gcc-12.1.0
mamba env create -f environment.yml
conda activate venv
```

Set HuggingFace cache to scratch storage before model downloads:

```bash
export HF_HOME=/scratch/$USER/hf_models
```

## Training Commands

```bash
sbatch scripts/rl.sh generate
sbatch scripts/rl.sh sft
sbatch scripts/rl.sh grpo
sbatch scripts/rl.sh pipeline
sbatch scripts/rl.sh all
```

`scripts/rl.sh` accepts `generate`, `sft`, `grpo`, `pipeline`, and `all`.

## Required Checkpoints

The repo does not commit model checkpoints. Before running GRPO or evaluation,
confirm these paths exist on SOL or adjust the YAML configs:

```text
checkpoints/sft-tuned-2
checkpoints/grpo-annotated-0411-1/checkpoint-1200
checkpoints/grpo-selfplay-0413-2
```

Relevant configs:

```text
rl/configs/sft.yaml
rl/configs/grpo.yaml
eval/config.yaml
eval/negotiate.yaml
```

## Evaluation Commands

Task-style evaluation:

```bash
python -m eval --list-tasks
python -m eval
python -m eval --evaluate-only
```

Head-to-head negotiation evaluation:

```bash
python -m eval.negotiate --config eval/negotiate.yaml
```

SLURM wrappers:

```bash
mkdir -p logs/jobs
sbatch scripts/eval.sh
sbatch scripts/eval.sh list
sbatch scripts/eval.sh score
sbatch scripts/negotiate_eval.sh eval/negotiate.yaml
```

## Final Submission Checks

- README commands match script arguments.
- `logs/jobs/` exists before submitting SLURM jobs.
- Checkpoint paths referenced in YAML exist on SOL.
- `uv run -m eval --list-tasks` works locally.
- The report PDF was regenerated after edits to `report/report.md`.
- Large generated outputs stay out of git: `logs/`, `runs/`, `raw/`, and
  `checkpoints/`.
