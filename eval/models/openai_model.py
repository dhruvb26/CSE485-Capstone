"""
OpenAI model handler (GPT-4o, GPT-4o-mini, etc.).

Uses the OpenAI chat completions API.
"""

from openai import OpenAI
import os
from tqdm import tqdm
from .base import BaseModelHandler
import tiktoken
import numpy as np

openai_api_key = os.getenv("OPENAI_API_KEY")


class OpenAIHandler(BaseModelHandler):

    multishot = False
    cot = False
    token_limit = 4096

    def setup_model(self):
        self.model = self.args.openai_model_str
        self.client = OpenAI(api_key=openai_api_key)

    def clean_prompt(self, prompt):
        if not isinstance(prompt, str):
            return prompt
        try:
            cleaned = prompt.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
            return cleaned
        except Exception as e:
            print(f"Warning: Failed to clean prompt: {e}")
            return prompt

    def check_prompt(self, prompt):
        encoding = tiktoken.get_encoding('cl100k_base')
        encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
        list_of_tokens = encoding.encode(prompt)
        num_tokens = len(list_of_tokens)
        return num_tokens < self.token_limit, num_tokens

    def process_gt(self, ppt, gt):
        if isinstance(gt, str):
            return gt
        if isinstance(gt, list):
            if "Present your answer as a comma-separated list of strategies" in ppt:
                return f"<answer>{', '.join(gt)}</answer>"
            elif "Present your answer as a Python list of the relevant options" in ppt:
                return f"```python\ndialogue_acts = {gt}\n```"
            else:
                raise ValueError
        raise ValueError

    def get_multishot_exs(self, inputs, ground_truth, index):
        assert len(inputs) == len(ground_truth)
        all_ixs = list(range(len(inputs)))
        all_ixs.remove(index)
        ix1, ix2 = np.random.choice(all_ixs, 2, replace=False)
        ex_prompt, ex_ans = inputs[ix1], ground_truth[ix1]
        sec_ex_prompt, sec_ex_ans = inputs[ix2], ground_truth[ix2]
        ex_ans_proc = self.process_gt(ex_prompt, ex_ans)
        sec_ex_ans_proc = self.process_gt(sec_ex_prompt, sec_ex_ans)
        return ex_prompt, ex_ans_proc, sec_ex_prompt, sec_ex_ans_proc

    def get_model_outputs(self, inputs, ground_truth):
        outputs = {}

        if self.multishot:
            ex_prompt = inputs[0]
            sec_ex_prompt = inputs[1]
            ex_ans = str(ground_truth[0]) if not isinstance(ground_truth[0], str) else ground_truth[0]
            sec_ex_ans = str(ground_truth[1]) if not isinstance(ground_truth[1], str) else ground_truth[1]

            for index in range(2, len(inputs)):
                prompt = "User:\n" + ex_prompt + "\n\nAssistant: " + ex_ans + "\n\nUser:\n" + sec_ex_prompt + "\n\nAssistant: " + sec_ex_ans + "\n\nUser:\n" + inputs[index] + "\n\nAssistant:"
                prompt = self.clean_prompt(prompt)
                assert self.check_prompt(prompt)

                gen_output = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                output_text = gen_output.choices[0].message.content
                outputs[prompt] = output_text
        else:
            for index in tqdm(range(len(inputs))):
                if self.args.num_multishot == 0:
                    prompt = "User:\n" + inputs[index] + "\n\nAssistant:"
                else:
                    assert self.args.num_multishot == 2
                    ex_prompt, ex_ans, sec_ex_prompt, sec_ex_ans = self.get_multishot_exs(inputs, ground_truth, index)
                    prompt = "User:\n" + ex_prompt + "\n\nAssistant: " + ex_ans + "\n\nUser:\n" + sec_ex_prompt + "\n\nAssistant: " + sec_ex_ans + "\n\nUser:\n" + inputs[index] + "\n\nAssistant:"

                outputs[inputs[index]] = "looks good."

                try:
                    prompt = self.clean_prompt(prompt)
                    a, b = self.check_prompt(prompt)
                    if not a:
                        print(f"Length issue at {index}/{len(inputs)}: {b} > {self.token_limit}")
                        continue

                    gen_output = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0
                    )
                    output_text = gen_output.choices[0].message.content
                    outputs[inputs[index]] = output_text
                except Exception as e:
                    print(f"Error at index {index}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                if (index + 1) >= self.args.max_num_instances:
                    break

        return outputs
