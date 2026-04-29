# Evaluation

Two evaluation modes: **tasks** (comprehension/generation benchmarks) and **negotiate** (self-play dialogue).

```
eval/
├── __main__.py          # CLI dispatcher: `python -m eval tasks|negotiate`
├── main.py              # Tasks evaluation orchestrator
├── registry.py          # (dataset, model, task) config registry
├── metrics.py           # Scoring functions (accuracy, F1, BLEU/ROUGE)
├── display.py           # Rich console output
├── utils.py             # Shared helpers
├── configs/
│   ├── tasks.yaml       # Config for task evaluation
│   └── negotiate.yaml   # Config for self-play negotiation
├── models/
│   ├── base.py          # BaseModelHandler ABC
│   ├── openai_model.py  # OpenAI API (GPT-4o, etc.)
│   ├── local_model.py   # Local checkpoints + LoRA merging
│   └── vllm_model.py    # vLLM OpenAI-compatible API
├── negotiate/
│   ├── envs.py          # NegotiateEnv ABC + 5 dataset environments
│   ├── prompts.py       # System prompts for all negotiation environments
│   └── harness.py       # Self-play loop, scoring, logging
├── tasks/               # Per-task handlers (SysEval benchmark)
└── datasets/            # Dataset loaders (DnD, CaSiNo, JI, CRA)
```

The tasks side is adapted from [SysEval-NegoLLMs](https://github.com/DSincerity/SysEval-NegoLLMs):

> Deuksin Kwon, Emily Weiss, Tara Kulshrestha, Kushal Chawla, Gale Lucas, and Jonathan Gratch. 2024. **Are LLMs Effective Negotiators? Systematic Evaluation of the Multifaceted Capabilities of LLMs in Negotiation Dialogues.** In *Findings of the Association for Computational Linguistics: EMNLP 2024*, pages 5391–5413.

## Task Evaluation

Runs comprehension and generation tasks across 4 datasets and 3 dialogue stages (start, mid, end).

```bash
python -m eval tasks --config eval/configs/tasks.yaml
python -m eval tasks --evaluate-only     # score existing logs only
python -m eval tasks --list-tasks        # show all available tasks
```

### Config

```yaml
models:
  - type: local_model
    model_path: checkpoints/grpo-tuned
    base_model: Qwen/Qwen2.5-7B-Instruct
    label: my-grpo-model
    max_new_tokens: 2048

tasks:
  - all_dnd           # all DnD tasks
  # or list individual tasks:
  # - sta_total_item_count_dnd
  # - end_deal_specifics_ca

num_instances: 200
use_cot: true
storage_dir: ./logs/eval
```

### Supported Models

| Type | Config Key | Notes |
|------|-----------|-------|
| **OpenAI** | `open_ai` | GPT-4o, GPT-4o-mini, etc. Needs `OPENAI_API_KEY` |
| **Local** | `local_model` | Any AutoModelForCausalLM checkpoint, auto-detects LoRA |
| **vLLM** | `vllm_model` | OpenAI-compatible API from `vllm serve` |

### Task Shortcuts

```yaml
tasks:
  - all               # every task across all datasets
  - all_dnd           # all Deal-or-No-Deal tasks
  - all_casino        # all CaSiNo tasks
  - all_job_interview # all Job Interview tasks
  - all_cra           # all CRA tasks
```

## Self-Play Negotiation

Runs LLM-vs-LLM negotiation dialogues, scores deals, and computes metrics.

```bash
python -m eval negotiate --config eval/configs/negotiate.yaml
python -m eval negotiate --evaluate-only logs/negotiate  # rescore existing logs
```

### Config

```yaml
dataset: casino              # casino | dnd | amazon | craigslist | ji
csv_path: data/casino/ca.test.csv
num_episodes: 100
max_turns: 12
output_dir: logs/negotiate

matchups:
  - learner:
      type: api
      base_url: http://localhost:8000/v1
      model: Qwen/Qwen3-4B-Instruct-2507
    opponent:
      type: api
      base_url: https://tinker.thinkingmachines.dev/...
      model: Qwen/Qwen3-30B-A3B-Instruct-2507
```

### Supported Datasets

| Dataset | Type | Agent IDs |
|---------|------|-----------|
| **CaSiNo** | Multi-item integrative (food/water/firewood) | Participant IDs from CSV |
| **DnD** | Multi-item integrative (book/hat/ball) | `agent_0`, `agent_1` |
| **AmazonHistoryPrice** | Single-issue price (buyer/seller) | `buyer`, `seller` |
| **Craigslist Bargains** | Single-issue price (buyer/seller) | `buyer`, `seller` |
| **Job Interview** | Multi-attribute hybrid (5 issues) | `worker`, `recruiter` |

## Environment Variables

```bash
export OPENAI_API_KEY=...    # OpenAI models
export TINKER_API_KEY=...    # Tinker API (ASU)
```

## Output

Logs are saved as JSON under the configured `storage_dir` / `output_dir`:

```
logs/eval/     # task evaluation results
logs/negotiate/ # self-play dialogue transcripts + scores
```
