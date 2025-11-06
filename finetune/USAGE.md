## format_dataset.py

Converts raw negotiation datasets into instruction-tuning formats (Alpaca, ChatML, ShareGPT).

### Key Features

### Usage

```bash
python format_dataset.py \
  --data-dir data \
  --dataset-name craigslist_bargains \
  --platform-name "Craigslist" \
  --max-samples 1000 \
  --formats alpaca
```

> **NOTE:** Run the script with `--help` for more information.

### Format Differences & Purposes

- **Alpaca**: Instruction-following format (`instruction`/`input`/`output`)

  - Purpose: Supervised fine-tuning for specific response generation
  - Structure: Separates "what others said" (input) vs "what you say" (output)
  - Output: 2 samples per conversation (buyer + seller perspectives)

- **ChatML**: Conversational format (`messages` array with `role`/`content`)

  - Purpose: Multi-turn conversation training
  - Structure: Maintains full conversational flow with system/user/assistant roles
  - Output: 2 samples per conversation (buyer + seller perspectives)

- **ShareGPT**: Dialogue format (`conversations` array with `from`/`value`)
  - Purpose: Training conversational AI models (ShareGPT-style datasets)
  - Structure: Complete dialogue sequence showing all participants
  - Output: 1 sample per conversation (full dialogue)

Files saved as: `{dataset_name}_{format}.jsonl` in the output directory.
