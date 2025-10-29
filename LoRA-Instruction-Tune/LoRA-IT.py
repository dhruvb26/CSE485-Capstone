#LoRA fine-tuning is used for quick training and experimenting

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import TrainingArguments, Trainer
import torch
from datasets import load_dataset

#Configs
model_name = "Qwen/Qwen2.5-7B-Chat" #edit based on Model to Tune
dataset = load_dataset("json", data_files="casino_instrcution_tuning.jsonl")["train"] #edit dataset path based on which dataset is being used for instruction tuning
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
) #Edit LoRA Config when Needed depending on model
training_args = TrainingArguments(
    output_dir="./qwen2.5-bargain-lora",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,               # use BF16 on A100
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
)


#load model and tokenizer
def load_model_tokenizer() -> list:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype="bfloat16"
    )

    return [model, tokenizer]

#Correctly Format Data for Instruction Tuning, This will be implemented differently for each dataset
#example format
def format_example(example, tokenizer):
    prompt = (
        f"Instruction: {example['instruction']}\n"
        f"Input: {example['input']}\n"
        f"Response:"
    )
    text = f"{prompt} {example['output']}"
    return tokenizer(text, truncation=True, padding="max_length", max_length=1024)



def train():
    #Load Model and Tokenizer
    model, tokenizer = load_model_tokenizer()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    #Load tokenized dataset
    tokenized_dataset = dataset.map(format_example, batched=False)

    #Declare trainer with arguments
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    #train
    trainer.train()
    model.save_pretrained("qwen2.5-bargain-lora") #contains only LoRA adapter weights and not base model
    tokenizer.save_pretrained("qwen2.5-bargain-lora")

#Smoke Check
def test():
    base_model = "Qwen/Qwen2.5-7B-Chat"
    adapter_model = "qwen2.5-bargain-lora"

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Base (untuned)
    base = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.bfloat16)

    # Tuned
    lora = PeftModel.from_pretrained(base, adapter_model)
    lora.eval()

    prompt = "Partner: I want the tent and sleeping bag."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = lora.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))