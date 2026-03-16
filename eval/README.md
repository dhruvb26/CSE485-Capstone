# Evaluation

This module provides a YAML-driven evaluation framework for systematically assessing LLM capabilities in negotiation dialogues. It is adapted from the [SysEval-NegoLLMs](https://github.com/DSincerity/SysEval-NegoLLMs) codebase accompanying the following paper:

> Deuksin Kwon, Emily Weiss, Tara Kulshrestha, Kushal Chawla, Gale Lucas, and Jonathan Gratch. 2024. **Are LLMs Effective Negotiators? Systematic Evaluation of the Multifaceted Capabilities of LLMs in Negotiation Dialogues.** In *Findings of the Association for Computational Linguistics: EMNLP 2024*, pages 5391–5413. Association for Computational Linguistics.

```bibtex
@inproceedings{kwon-etal-2024-llms,
    title     = "Are {LLM}s Effective Negotiators? Systematic Evaluation of the Multifaceted Capabilities of {LLM}s in Negotiation Dialogues",
    author    = "Kwon, Deuksin and Weiss, Emily and Kulshrestha, Tara and Chawla, Kushal and Lucas, Gale and Gratch, Jonathan",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    year      = "2024",
    publisher = "Association for Computational Linguistics",
    pages     = "5391--5413",
    url       = "https://arxiv.org/abs/2402.13550"
}
```

## Quick Start

### 1. Configure your evaluation

Edit `eval/config.yaml`:

```yaml
models:
  - type: open_ai
    model_str: gpt-4o-mini-2024-07-18

tasks:
  - sta_total_item_count_dnd
  - end_deal_specifics_ca

num_instances: 200
storage_dir: ./logs/eval
```

### 2. Run evaluation

```bash
# Run with default config
python -m eval

# Run with custom config
python -m eval --config path/to/my_config.yaml

# Only score existing logs (no model inference)
python -m eval --evaluate-only

# List all available tasks
python -m eval --list-tasks
```

### 3. Environment variables

For OpenAI models, set:
```bash
export OPENAI_API_KEY=your-key-here
```

## Available Tasks

Tasks span **4 negotiation datasets** and **3 dialogue stages** (start, mid, end):

| Dataset | Code | Description |
|---------|------|-------------|
| **Deal or No Deal** | `dnd` | Book/hat/ball item negotiation |
| **CaSiNo** | `casino` / `ca` | Campsite food/water/firewood negotiation |
| **Job Interview** | `job_interview` / `ji` | 5-issue job offer negotiation |
| **CRA** | `cra` | Painting/lamp/record negotiation |

### Task Categories

| Stage | Task Type | Example Tasks | Metric |
|-------|-----------|---------------|--------|
| **Start** | Comprehension | `sta_total_item_count_dnd`, `sta_max_points_ca` | Accuracy |
| **Start** | Point values | `sta_ask_point_values_dnd`, `sta_ask_point_values_ca` | Elementwise Accuracy |
| **Start** | Priorities | `sta_ask_high_priority_ca`, `sta_ask_low_priority_ji_w` | Accuracy |
| **Mid** | Dialog acts | `mid_dial_act_dnd`, `mid_dial_act_cra` | Macro-F1 |
| **Mid** | Strategy | `mid_strategy_ca` | Macro-F1 |
| **Mid** | Generation | `mid_gen_resp_dnd`, `mid_gen_resp_ca` | BLEU/ROUGE |
| **Mid** | Proposals | `mid_full_proposal_dnd`, `mid_full_proposal_cra` | Elementwise Accuracy |
| **Mid** | Partner modeling | `mid_partner_ask_high_priority_ca` | Accuracy |
| **End** | Deal specifics | `end_deal_specifics_dnd`, `end_deal_specifics_ca` | Elementwise Accuracy |
| **End** | Deal totals | `end_deal_total_dnd`, `end_deal_total_ca` | Accuracy |
| **End** | Subjective | `end_deal_likeness_ca`, `end_deal_satisfaction_ca` | Accuracy |

### Shortcuts

In the YAML config, use these shortcuts:

```yaml
tasks:
  - all              # every task across all datasets
  - all_dnd          # all Deal-or-No-Deal tasks
  - all_casino       # all CaSiNo tasks
  - all_job_interview # all Job Interview tasks
  - all_cra          # all CRA tasks
```

## Supported Models

| Type | Config Key | Examples |
|------|-----------|----------|
| **OpenAI** | `open_ai` | `gpt-4o-mini-2024-07-18`, `gpt-4o-2024-08-06`, `gpt-3.5-turbo` |
| **HuggingFace** | `hf_model` | `mistralai/Mistral-7B-Instruct-v0.1`, `google/flan-t5-base` |
| **Local / Custom** | `local_model` | Any `AutoModelForCausalLM`-compatible checkpoint, including LoRA adapters |

### Using custom-trained models

The `local_model` type loads any checkpoint produced by your training pipeline (SFT, GRPO, etc.):

```yaml
models:
  - type: local_model
    model_path: checkpoints/grpo-tuned       # checkpoint dir or HF hub name
    base_model: Qwen/Qwen2.5-3B-Instruct    # required if model_path is a LoRA adapter
    label: my-grpo-model                     # optional display name for log files
    max_new_tokens: 256                      # generation length (default: 256)
    token_limit: 4096                        # max input tokens (default: 4096)
```

- **Full checkpoint**: if `model_path` contains a full model, only `model_path` is needed.
- **LoRA adapter**: if `adapter_config.json` is found in `model_path`, the handler loads `base_model` first, merges the adapter, then runs inference.
- Prompts are formatted using the tokenizer's `apply_chat_template` when available.

## Configuration Reference

```yaml
models:
  - type: open_ai                           # "open_ai", "hf_model", or "local_model"
    model_str: gpt-4o-mini-2024-07-18       # model identifier

  - type: local_model
    model_path: checkpoints/grpo-tuned      # path to checkpoint or HF hub name
    base_model: Qwen/Qwen2.5-3B-Instruct   # base model for LoRA adapters
    label: my-grpo-model                    # display name (defaults to dir basename)

tasks:
  - sta_total_item_count_dnd

num_instances: 200          # test instances per task (max 200)
max_num_instances: 200      # hard ceiling

use_cot: false              # chain-of-thought
num_multishot: 0            # few-shot examples (0 or 2)
num_prior_utts: 0           # prior context utterances
num_utts_partial_dial: -1   # partial dialogue (-1 = full)

storage_dir: ./logs/eval    # output directory (gitignored via logs/)
evaluate_only: false        # if true, just score existing logs
```

## Output

Evaluation logs are saved as JSON files under `logs/eval/`:

```
logs/eval/
  dnd_gpt-4o-mini-2024-07-18_sta_total_item_count_dnd_200.json
  casino_gpt-4o-mini-2024-07-18_end_deal_specifics_ca_200.json
  dnd_my-grpo-model_sta_total_item_count_dnd_200.json
  ...
```

Each log contains:
- `ground truth`: expected answers
- `predictions`: model predictions
- `prompts`: prompts sent to the model
- `outputs_dict`: raw model outputs
- `stats`: instance counts (total, unique, valid)
