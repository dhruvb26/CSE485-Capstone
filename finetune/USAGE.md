## Format Data

```bash
python format_dataset.py \
  --data-dir ../data \
  --dataset-name craigslist_bargains \
  --platform-name "Craigslist" \
  --output-dir formatted_data \
  --split train \
  --max-samples 1000 \
  --formats alpaca
```

Quick check:

```bash
head -n 2 formatted_data/craigslist_bargains_alpaca.jsonl | jq .
```

> **NOTE:** Run the script with `--help` for more information.

### Format Differences & Purposes

- **Alpaca**: Instruction-following format (`instruction`/`input`/`output`)

  - Purpose: Supervised fine-tuning for specific response generation
  - Structure: Separates "what others said" (input) vs "what you say" (output)
  - Output: 2 samples per conversation (buyer + seller perspectives)

- **chatml**: Conversational format (`messages` array with `role`/`content`)

  - Purpose: Multi-turn conversation training
  - Structure: Maintains full conversational flow with system/user/assistant roles
  - Output: 2 samples per conversation (buyer + seller perspectives)

- **ShareGPT**: Dialogue format (`conversations` array with `from`/`value`)
  - Purpose: Training conversational AI models (ShareGPT-style datasets)
  - Structure: Complete dialogue sequence showing all participants
  - Output: 1 sample per conversation (full dialogue)

Files saved as: `{dataset_name}_{format}.jsonl` in the output directory.

---

## Fine tune Gwen

```bash
python gwen.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset finetune/formatted_data/craigslist_bargains_alpaca.jsonl \
  --mode train
```

Adapter files should be located in `/gwen_adapter_v0`

- [TODO] Post evals
