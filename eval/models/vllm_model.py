"""
vLLM model handler — uses the OpenAI-compatible API exposed by ``vllm serve``.

Start the server first (see scripts/vllm_eval.sh), then point this handler
at it via config:

    - type: vllm_model
      model_str: Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled
      base_url: http://localhost:8000/v1
      max_tokens: 512
      label: qwen3.5-9b-reasoning
"""

import os
import re
from tqdm import tqdm
from .base import BaseModelHandler

from openai import OpenAI


class VLLMModelHandler(BaseModelHandler):

    multishot = False
    cot = False

    def setup_model(self):
        self.model = getattr(self.args, "openai_model_str", "")
        base_url = getattr(self.args, "base_url", "http://localhost:8000/v1")
        api_key = os.getenv("OPENAI_API_KEY", "dummy")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.max_tokens = getattr(self.args, "max_tokens", 512)
        self.token_limit = getattr(self.args, "token_limit", 8192)
        self._label = getattr(self.args, "label", None)
        self.cot = getattr(self.args, "use_cot", False)

    @property
    def display_name(self):
        if self._label:
            return self._label
        return self.model.rsplit("/", 1)[-1]

    def check_prompt(self, text):
        n = len(text) // 4
        return n < self.token_limit, n

    def get_model_outputs(self, inputs, ground_truth):
        outputs = {}

        for index in tqdm(range(len(inputs))):
            raw_prompt = inputs[index]
            prompt = raw_prompt.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

            ok, n_est = self.check_prompt(prompt)
            if not ok:
                print(f"Length issue at {index}/{len(inputs)}: ~{n_est} tokens > {self.token_limit}")
                continue

            try:
                gen_output = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=0,
                )
                output_text = gen_output.choices[0].message.content or ""
                output_text = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL).strip()
                outputs[inputs[index]] = output_text
            except Exception as e:
                print(f"Error at index {index}: {e}")
                continue

            if len(outputs) >= self.args.max_num_instances:
                break

        return outputs
