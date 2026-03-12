"""Model handler that loads a GRPO-trained model for SysEval evaluation."""

from __future__ import annotations

import logging
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class GRPOModelHandler:
    """SysEval-compatible model handler that wraps a GRPO-trained checkpoint.

    Implements the same interface as SysEval's ``BaseModelHandler``:
    ``setup_model()`` and ``get_model_outputs(inputs, ground_truth)``.

    The handler loads the base model, optionally merges the SFT LoRA
    adapter, then optionally merges a GRPO LoRA adapter on top. Prompts
    from SysEval task handlers are wrapped into the model's chat template
    before generation.
    """

    multishot = False
    cot = False

    def __init__(self, name, args, eval_config):
        self.name = name
        self.args = args
        self._eval_config = eval_config
        self.max_new_tokens = 256
        self.token_limit = 4096
        self.setup_model()

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

    def get_model_outputs(self, inputs, ground_truth):
        """Generate outputs for a list of text prompts.

        Each prompt is wrapped as a single user message in the chat
        template, then decoded greedily (temperature=0).

        Returns a dict mapping ``{prompt_text: output_text}``.
        """
        outputs = {}

        for prompt_text in tqdm(inputs, desc="Generating"):
            messages = [{"role": "user", "content": prompt_text}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self.tokenizer(
                text, return_tensors="pt"
            ).to(self.model.device)

            if model_inputs.input_ids.shape[1] > self.token_limit:
                log.warning(
                    "Prompt too long (%d tokens > %d limit), skipping",
                    model_inputs.input_ids.shape[1],
                    self.token_limit,
                )
                continue

            with torch.no_grad():
                generated = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            new_tokens = generated[0, model_inputs.input_ids.shape[1]:]
            output_text = self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()
            outputs[prompt_text] = output_text

            if len(outputs) >= self.args.max_num_instances:
                break

        return outputs
