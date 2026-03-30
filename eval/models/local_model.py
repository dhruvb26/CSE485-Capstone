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
    trust_remote_code: false                   # Hub models that need custom code
    max_new_tokens: 256
    label: my-grpo-model                       # optional display name for logs
"""

import os
import re
from tqdm import tqdm
from .base import BaseModelHandler


def _load_pretrained_lm(model_path, trust_remote_code=False):
    """
    Load a text-generation model from a checkpoint or Hub id.

    Qwen3.5 VL checkpoints (Qwen3_5ForConditionalGeneration) are not
    AutoModelForCausalLM; we load them explicitly for text-only eval.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    load_kw = {"device_map": "auto", "dtype": "auto", "trust_remote_code": trust_remote_code}
    arch = None
    try:
        cfg = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code,
        )
        if getattr(cfg, "architectures", None):
            arch = cfg.architectures[0]
    except Exception:
        pass

    if arch == "Qwen3_5ForConditionalGeneration":
        try:
            from transformers import Qwen3_5ForConditionalGeneration
        except ImportError as e:
            raise ImportError(
                "This checkpoint is Qwen3_5ForConditionalGeneration; install a "
                "recent transformers (e.g. >= 4.49) that provides this class."
            ) from e
        model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **load_kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kw)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code,
    )
    return model, tokenizer


class LocalModelHandler(BaseModelHandler):

    multishot = False
    cot = False

    def setup_model(self):
        model_path = getattr(self.args, "model_path", None)
        base_model = getattr(self.args, "base_model", None)
        if not model_path:
            raise ValueError("local_model requires 'model_path' in the YAML config")

        self.max_new_tokens = getattr(self.args, "max_new_tokens", 256)
        self.token_limit = getattr(self.args, "token_limit", 4096)
        self._label = getattr(self.args, "label", None)
        self.cot = getattr(self.args, "use_cot", False)
        trust_remote_code = getattr(self.args, "trust_remote_code", False)

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
            self.model, self.tokenizer = _load_pretrained_lm(
                base_model, trust_remote_code=trust_remote_code,
            )

            print(f"  Merging LoRA adapter: {model_path}")
            self.model = PeftModel.from_pretrained(self.model, model_path)
            self.model = self.model.merge_and_unload()
        else:
            print(f"  Loading model from: {model_path}")
            self.model, self.tokenizer = _load_pretrained_lm(
                model_path, trust_remote_code=trust_remote_code,
            )

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
            output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
            output_text = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL).strip()
            output_text = re.sub(r"<thought>.*?</thought>", "", output_text, flags=re.DOTALL).strip()
            output_text = re.sub(r"<talk>.*?</talk>", "", output_text, flags=re.DOTALL).strip()
            for tok in self.tokenizer.all_special_tokens:
                output_text = output_text.replace(tok, "")
            output_text = output_text.strip()

            outputs[inputs[index]] = output_text

            if len(outputs) >= self.args.max_num_instances:
                break

        return outputs
