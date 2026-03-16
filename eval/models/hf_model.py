"""
HuggingFace model handler for open-source models.

Supports: Flan-T5, Falcon, Mistral, Vicuna, WizardLM.
"""

from tqdm import tqdm
from .base import BaseModelHandler


class HFModelHandler(BaseModelHandler):

    multishot = False
    cot = False

    def setup_model(self):
        if "flan-t5" in self.args.hf_model_str:
            self.token_limit = 512
            self.max_new_tokens = 100
        elif "falcon" in self.args.hf_model_str:
            self.token_limit = 1500
            self.max_new_tokens = 2048
        elif "mistral" in self.args.hf_model_str:
            self.token_limit = 1500
            self.max_new_tokens = 200
        elif "vicuna" in self.args.hf_model_str or "Wizard" in self.args.hf_model_str:
            self.token_limit = 1500
            self.max_new_tokens = 200
        else:
            raise ValueError(f"Unsupported HF model: {self.args.hf_model_str}")

        if "flan-t5" in self.args.hf_model_str:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.hf_model_str, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_model_str)
        elif "falcon" in self.args.hf_model_str:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model = AutoModelForCausalLM.from_pretrained(self.args.hf_model_str, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_model_str)
        elif "mistral" in self.args.hf_model_str:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model = AutoModelForCausalLM.from_pretrained(self.args.hf_model_str, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_model_str)
        elif "vicuna" in self.args.hf_model_str or "Wizard" in self.args.hf_model_str:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model = AutoModelForCausalLM.from_pretrained(self.args.hf_model_str, device_map="auto", low_cpu_mem_usage=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_model_str)
        else:
            raise ValueError(f"Unsupported HF model: {self.args.hf_model_str}")

    def check_prompt(self, prompt):
        if "mistral" in self.args.hf_model_str:
            return True, 0
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").input_ids
            return inputs.shape[1] < self.token_limit, inputs.shape[1]

    def get_model_outputs(self, inputs, ground_truth):
        outputs = {}

        if self.multishot or self.cot:
            raise NotImplementedError

        for index in tqdm(range(len(inputs))):
            if "flan-t5" in self.args.hf_model_str:
                prompt = inputs[index]
            elif "mistral" in self.args.hf_model_str:
                prompt = [{"role": "user", "content": inputs[index]}]
            else:
                prompt = "User: " + inputs[index] + " Assistant: "

            a, b = self.check_prompt(prompt)
            if not a:
                print(f"Length issue at {index}/{len(inputs)}: {b} > {self.token_limit}")
                continue

            outputs[inputs[index]] = self._get_hf_output(prompt)

            if len(outputs) >= self.args.max_num_instances:
                break

        return outputs

    def _get_hf_output(self, prompt):
        if "flan" in self.args.hf_model_str:
            inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            outputs = self.model.generate(inputs, max_new_tokens=self.max_new_tokens)
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        elif "falcon" in self.args.hf_model_str:
            inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            outputs = self.model.generate(inputs, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt):]
        elif "mistral" in self.args.hf_model_str:
            text = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            if not isinstance(text, str):
                text = str(text)
            text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
            model_inputs = self.tokenizer(text, return_tensors="pt").input_ids.to("cuda")
            generated_ids = self.model.generate(model_inputs, max_new_tokens=self.max_new_tokens)
            decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            return decoded[0].split("[/INST]")[-1].strip()
        elif "vicuna" in self.args.hf_model_str or "Wizard" in self.args.hf_model_str:
            inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            outputs = self.model.generate(inputs, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt):]

        raise ValueError(f"Unsupported HF model: {self.args.hf_model_str}")
