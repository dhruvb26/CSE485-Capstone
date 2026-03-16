"""
Local model handler for custom-trained checkpoints.

Supports:
  - Full model checkpoints (any AutoModelForCausalLM-compatible directory)
  - LoRA adapter checkpoints (auto-detected via adapter_config.json, merged on load)
  - Any HuggingFace hub model (as base or standalone)

Uses apply_chat_template for prompt formatting when available.

Config example:
  - type: local_model
    model_path: checkpoints/grpo-tuned         # checkpoint dir or HF hub name
    base_model: Qwen/Qwen2.5-3B-Instruct      # required for LoRA adapters
    max_new_tokens: 256
    label: my-grpo-model                       # optional display name for logs
"""

import os
from tqdm import tqdm
from .base import BaseModelHandler


class LocalModelHandler(BaseModelHandler):

    multishot = False
    cot = False

    def setup_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = getattr(self.args, "model_path", None)
        base_model = getattr(self.args, "base_model", None)
        if not model_path:
            raise ValueError("local_model requires 'model_path' in the YAML config")

        self.max_new_tokens = getattr(self.args, "max_new_tokens", 256)
        self.token_limit = getattr(self.args, "token_limit", 4096)
        self._label = getattr(self.args, "label", None)

        adapter_config = os.path.join(model_path, "adapter_config.json")
        is_lora = os.path.exists(adapter_config)

        if is_lora:
            if not base_model:
                raise ValueError(
                    f"LoRA adapter detected at {model_path} but no 'base_model' specified. "
                    "Set base_model to the HF model the adapter was trained on."
                )
            from peft import PeftModel

            print(f"  Loading base model: {base_model}")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model, device_map="auto", dtype="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)

            print(f"  Merging LoRA adapter: {model_path}")
            self.model = PeftModel.from_pretrained(self.model, model_path)
            self.model = self.model.merge_and_unload()
        else:
            print(f"  Loading model from: {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", dtype="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def display_name(self):
        """Name used in log filenames."""
        if self._label:
            return self._label
        path = getattr(self.args, "model_path", "local")
        return os.path.basename(path.rstrip("/"))

    def check_prompt(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").input_ids
        n = inputs.shape[1]
        return n < self.token_limit, n

    def get_model_outputs(self, inputs, ground_truth):
        outputs = {}

        for index in tqdm(range(len(inputs))):
            raw_prompt = inputs[index]

            if self.tokenizer.chat_template:
                messages = [{"role": "user", "content": raw_prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            else:
                text = "User:\n" + raw_prompt + "\n\nAssistant:"

            text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

            ok, n_tokens = self.check_prompt(text)
            if not ok:
                print(f"Length issue at {index}/{len(inputs)}: {n_tokens} > {self.token_limit}")
                continue

            model_inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            generated = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new_tokens = generated[0][model_inputs["input_ids"].shape[1]:]
            output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            outputs[inputs[index]] = output_text

            if len(outputs) >= self.args.max_num_instances:
                break

        return outputs
