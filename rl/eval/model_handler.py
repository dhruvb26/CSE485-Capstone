"""Model handler that loads a GRPO-trained model for SysEval evaluation."""

from __future__ import annotations

import json
import logging
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

EVAL_SYSTEM_PROMPT = (
    "You are a skilled negotiation agent in a camping supply negotiation "
    "involving food, water, and firewood packages. "
    "You will be given a negotiation scenario and asked a question about it.\n\n"
    "You may reason step-by-step inside <thought>...</thought> tags before "
    "answering. Then follow the question's instructions for your final answer "
    "format exactly."
)

_TAG_RE = re.compile(r"</?(?:thought|talk|action)>", re.IGNORECASE)
_THOUGHT_BLOCK_RE = re.compile(
    r"<thought>.*?</thought>\s*", re.DOTALL | re.IGNORECASE
)


class GRPOModelHandler:
    """SysEval-compatible model handler that wraps a GRPO-trained checkpoint.

    Implements the same interface as SysEval's ``BaseModelHandler``:
    ``setup_model()`` and ``get_model_outputs(inputs, ground_truth)``.

    The handler loads the base model, optionally merges the SFT LoRA
    adapter, then optionally merges a GRPO LoRA adapter on top. Prompts
    from SysEval task handlers are wrapped with a negotiation-aware system
    message that encourages ``<thought>`` reasoning before the final answer.
    Tags are stripped in post-processing so SysEval scorers see only the
    bare answer.
    """

    multishot = False
    cot = False

    def __init__(self, name, args, eval_config, completions_path: str | None = None):
        self.name = name
        self.args = args
        self._eval_config = eval_config
        self.max_new_tokens = 512
        self.token_limit = 4096
        self._completions_path = completions_path
        self._completions: list[dict] = []
        self._current_task: str = ""
        self.setup_model()

    def set_current_task(self, task_name: str):
        self._current_task = task_name

    def setup_model(self):
        cfg = self._eval_config
        model_cfg = cfg.model

        if model_cfg.hf_home:
            os.environ["HF_HOME"] = model_cfg.hf_home

        dtype = DTYPE_MAP.get(model_cfg.dtype, torch.float16)

        log.warning("Loading base model %s", model_cfg.name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_cfg.name,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.name)
        self.tokenizer.padding_side = "left"

        if cfg.sft_checkpoint and os.path.isdir(cfg.sft_checkpoint):
            self._merge_adapter(cfg.sft_checkpoint, dtype, label="SFT")

        if cfg.grpo_checkpoint and os.path.isdir(cfg.grpo_checkpoint):
            self._merge_adapter(cfg.grpo_checkpoint, dtype, label="GRPO")

        self.model.eval()

    def _merge_adapter(self, ckpt_path: str, dtype, label: str):
        from peft import PeftModel

        adapter_cfg = os.path.join(ckpt_path, "adapter_config.json")
        if os.path.exists(adapter_cfg):
            log.warning("Loading %s LoRA adapter from %s", label, ckpt_path)
            self.model = PeftModel.from_pretrained(self.model, ckpt_path)
            self.model = self.model.merge_and_unload()
            log.warning("Merged %s adapter into base model", label)
        else:
            log.warning(
                "%s checkpoint %s has no adapter_config.json — "
                "loading as full model weights",
                label,
                ckpt_path,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                ckpt_path, torch_dtype=dtype, device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
            self.tokenizer.padding_side = "left"

    @staticmethod
    def _clean_output(text: str) -> str:
        """Extract the final answer from model output.

        Prefers text that appears *after* a ``</thought>`` block, falling
        back to stripping all tags if no thought block is present.
        """
        match = _THOUGHT_BLOCK_RE.search(text)
        if match:
            after = text[match.end():].strip()
            after = _TAG_RE.sub("", after).strip()
            if after:
                return after
        text = _THOUGHT_BLOCK_RE.sub("", text)
        text = _TAG_RE.sub("", text)
        return text.strip()

    def get_model_outputs(self, inputs, ground_truth):
        """Generate outputs for a list of text prompts.

        Each prompt is wrapped with a negotiation-aware system message that
        encourages chain-of-thought inside ``<thought>`` tags, then decoded
        greedily.  Tags are stripped before returning so SysEval scorers see
        only the bare answer.  Raw completions are accumulated for later
        saving via :meth:`save_completions`.

        Returns a dict mapping ``{prompt_text: cleaned_output_text}``.
        """
        outputs = {}

        for idx, prompt_text in enumerate(tqdm(inputs, desc="Generating")):
            prompt_str = str(prompt_text) if prompt_text is not None else ""
            messages = [
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_str},
            ]

            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                if isinstance(rendered, list):
                    rendered = rendered[0]
                # CaSiNo dialogues can contain lone Unicode surrogates that
                # the Rust tokenizer rejects.  Round-trip through bytes with
                # error replacement to produce clean UTF-8.
                rendered = rendered.encode("utf-8", errors="replace").decode("utf-8")
                encoding = self.tokenizer._tokenizer.encode(
                    rendered, add_special_tokens=False
                )
                ids = torch.tensor(
                    [encoding.ids], dtype=torch.long
                ).to(self.model.device)
            except Exception as exc:
                log.warning("Tokenization failed on prompt %d: %s", idx, exc)
                continue

            if ids.shape[1] > self.token_limit:
                log.warning(
                    "Prompt too long (%d tokens > %d limit), skipping",
                    ids.shape[1], self.token_limit,
                )
                continue

            with torch.no_grad():
                generated = self.model.generate(
                    ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            new_tokens = generated[0, ids.shape[1]:]
            raw = self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()
            cleaned = self._clean_output(raw)
            outputs[prompt_text] = cleaned

            gt_item = ground_truth[idx] if idx < len(ground_truth) else None
            self._completions.append({
                "task": self._current_task,
                "prompt": prompt_str[:500],
                "raw_completion": raw,
                "cleaned": cleaned,
                "ground_truth": gt_item,
            })

            if len(outputs) >= self.args.max_num_instances:
                break

        return outputs

    def save_completions(self):
        """Flush accumulated completions to the JSONL file."""
        if not self._completions_path or not self._completions:
            return
        os.makedirs(os.path.dirname(self._completions_path), exist_ok=True)
        with open(self._completions_path, "w") as f:
            for entry in self._completions:
                f.write(json.dumps(entry, default=str) + "\n")
        log.warning(
            "Saved %d completions to %s",
            len(self._completions),
            self._completions_path,
        )
