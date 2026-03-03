# GRPO Negotiation Training

Self-play reinforcement learning for buyer/seller negotiation agents using Group Relative Policy Optimization (GRPO).

---

## Overview

This project trains two LLM-based agents — a **Buyer** and a **Seller** — to negotiate product prices through self-play. Training uses a terminal reward signal derived from normalized profit metrics, with policy updates performed offline via a clipped GRPO objective and a KL-divergence penalty against a frozen reference model.

The system alternates between:
1. **Rollout phase** — running negotiations via a vLLM inference server
2. **Update phase** — loading model weights, computing GRPO loss, and saving a new checkpoint

---

## Project Structure

```
.
├── grpo.py                  # Main training script (this file)
├── agents.py                # BuyerAgent and SellerAgent definitions
├── clients.py               # LocalChat and OpenAIChat client wrappers
├── main.py                  # run_dialog and _get_client helpers
├── utils.py                 # Metrics, reward computation, vLLM lifecycle helpers
├── data/
│   └── amazon_history_price/ # Product dataset (JSON files)
├── logs/                    # Training logs (auto-created)
└── runs/                    # Per-session negotiation logs (auto-created)
```

---

## Requirements

- Python 3.10+
- PyTorch with CUDA
- `transformers`
- `bitsandbytes`
- `vllm`

---

## Configuration

Key parameters at the top of `grpo.py` and in `__main__`:

| Parameter | Default | Description |
|---|---|---|
| `product_limit` | `6` | Number of products to iterate over during training |
| `update_every` | `64` | Minimum buffer size before triggering a weight update |
| `load_every` | `2` | How often (in updates) to reload weights into vLLM |
| `save_path` | `/scratch/.../grpo_qwen_checkpoint` | Directory prefix for saved checkpoints |
| `cache_directory` | `/scratch/.../hf_cache_qwen/` | HuggingFace model cache directory |
| `current_weights_path` | `Qwen/Qwen2.5-7B-Instruct` | Initial model weights path |

Update these values before running, particularly `save_path` and `cache_directory`.

---

## Reward Signal

Rewards are based on **Normalized Profit (NP)** metrics computed at the end of each negotiation:

- **`NPb`** — Normalized profit for the buyer
- **`NPs`** — Normalized profit for the seller

Advantage for each agent is computed as:

```
advantage = reward - mean_reward_across_sessions
```

where the mean is taken over 8 rollouts per product.

---

## Training Loop

For each product:

1. Run **8 negotiation sessions** between buyer and seller copies
2. Extract per-turn `(query, response)` pairs and assign terminal advantages
3. Accumulate results into buyer and seller replay buffers
4. When buffer size exceeds `update_every`:
   - Stop the vLLM server
   - Load model + frozen reference model onto GPU
   - Run `grpo_update_offline` for buyer, then seller
   - Save checkpoint; optionally reload vLLM with updated weights

---

## Running

Load the venv as mentioned in ./README.md using...

```conda activate venv```

make sure the requirements in mentioned in this document are present within the venv, then run the script ...

```bash
python grpo.py
```

Logs are written to `logs/grpo_training_<timestamp>.log` and per-session negotiation logs to `runs/session_<timestamp>.log`.

---

## Checkpoints

Checkpoints are saved after each weight update to:

```
<save_path><update_count>/
```

Both model weights and tokenizer are saved. Set `load_every` to control how frequently the vLLM inference server picks up new weights during training.

---

## Notes

- The script assumes two GPUs: inference runs on `cuda:0` (vLLM) and training on `cuda:1`.
- `bitsandbytes` AdamW8bit is used to reduce optimizer memory overhead.
- Gradient checkpointing is enabled during the update phase.
- The vLLM server is stopped and restarted around each update to free VRAM.
