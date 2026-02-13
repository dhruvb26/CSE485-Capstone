import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict
import time
import copy
from trl import PPOTrainer, PPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


from agents import BuyerAgent, SellerAgent
from clients import LocalChat, OpenAIChat
from utils import (
    accumulate,
    compute_metrics,
    extract_action_and_price,
    finalize_aggregates,
    inventory_list,
    load_product,
    shopping_list,
)
from main import _get_client
from main import run_dialog

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)



'''
This GRPO Layout is to follow a simple terminal reward self play RL Design, this is an initial design and code implementation
'''

dataset_dir = 'data/amazon_history_price' #directory of location of dataset
product_limit = 10 #limit on number of products to train on 

class GRPO:


    def __init__(self, model1, model2, base_url):
        self.model1 = _get_client(model_name=model1, local_base_url=base_url)
        self.model1_name = model1
        self.model2 = _get_client(model_name=model2, local_base_url=base_url)
        self.model2_name = model2
        self.runs = {}
        self.run_incrementer = 0
        self.model1_tokenizer = AutoTokenizer.from_pretrained(model1, local_files_only=True)
        self.model2_tokenizer = AutoTokenizer.from_pretrained(model2, local_files_only=True)
        '''
        self.config = PPOConfig(
            model_name=model1,
            learning_rate=1.41e-5,
            batch_size=1,
            log_with=None)
        '''

    def add_run(self):
        self.run_incrementer += 1

#Helper function used to convert vllm (chat) style payload to actual model prompt
def chat_http_package_to_model_input(tokenizer, http_package):
    #extract messages field
    messages = http_package['messages']

    #format prompt from messages, tokenize
    prompt_tokens = tokenizer.apply_chat_template(messages, tokenize=True)

    # convert back to verify
    decoded = tokenizer.decode(prompt_tokens)

    return (prompt_tokens, decoded)

#Helper function used to convert vllm (chat) style payload to actual model output
def chat_http_package_from_model_output(tokenizer, http_package):
    # Extract the text from response
    response_text = http_package["choices"][0]["message"]["content"]

    # Tokenize it
    output_tokens = tokenizer.encode(response_text)

    # Convert back to verify
    decoded = tokenizer.decode(output_tokens)

    return (output_tokens, decoded)

#Helper function which takes an in order log of all http requests+responses made (model inputs+outputs) and seperates them into buyer (input, output) and seller (input, output) pairs
def extract_turns_from_all_messages(tokenizer_buyer, tokenizer_seller, negotiation_log):
    buyer_log_encoded = []
    seller_log_encoded = []
    buyer_log_decoded = []
    seller_log_decoded = []
    for index, input_output in enumerate(negotiation_log):
        input_package = input_output[0]
        output_package = input_output[1]
        if index % 2 == 0:
            input_encoded, input_decoded = chat_http_package_to_model_input(tokenizer=tokenizer_buyer, http_package=input_package)
            output_encoded, output_decoded = chat_http_package_from_model_output(tokenizer=tokenizer_buyer, http_package=output_package)
            buyer_log_encoded.append((input_encoded, output_encoded))
            buyer_log_decoded.append((input_decoded, output_decoded))
        else:
            input_encoded, input_decoded = chat_http_package_to_model_input(tokenizer=tokenizer_seller, http_package=input_package)
            output_encoded, output_decoded = chat_http_package_from_model_output(tokenizer=tokenizer_seller, http_package=output_package)
            seller_log_encoded.append((input_encoded, output_encoded))
            seller_log_decoded.append((input_decoded, output_decoded))
    
    return (buyer_log_encoded, seller_log_encoded, buyer_log_decoded, seller_log_decoded)




def run_single_grpo(GRPO, item, max_turns: int):
    
    #log file
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_log_file = runs_dir / f"session_{timestamp}.log"
    
    
    
    B = 0.8 * float(item["highest_price"])  # Come back to 0.8 value
    C = float(item["lowest_price"])
    inv_block = inventory_list(item)
    shop_block = shopping_list(item, B)
    buyer = BuyerAgent(
        client=GRPO.model1,
        model_name=GRPO.model1_name,
        inv_block=inv_block,
        shop_block=shop_block,
        B=B,
        code=item["codename"],
        max_turns=max_turns,
    )
    seller = SellerAgent(
        client=GRPO.model2,
        model_name=GRPO.model2_name,
        inv_block=inv_block,
        C=C,
        code=item["codename"],
        max_turns=max_turns,
    )

    outcomes = {}
    metrics = {}
    for i in range(8):
        buyer_copy = copy.deepcopy(buyer)
        seller_copy = copy.deepcopy(seller)
        outcome, negotiation_log = run_dialog(
            buyer=buyer_copy,
            seller=seller_copy,
            item=item,
            B=B,
            C=C,
            max_turns=max_turns,
            log_file=session_log_file,
            temperature= 0.5,
            top_p = 0.8)
        outcomes[i] = outcome
        m = compute_metrics(outcome, B, C)
        metrics[i] = m
    print(outcomes)
    print(metrics)
    buyer_log_encoded, seller_log_encoded, buyer_log_decoded, seller_log_decoded = extract_turns_from_all_messages(tokenizer_buyer=GRPO.model1_tokenizer, tokenizer_seller=GRPO.model2_tokenizer, negotiation_log=negotiation_log["allPackagesSentOrdered"])

    return outcomes, metrics

def run_single_ppo(trainer, inputs, outputs, rewards):
    for query, response, reward in zip(inputs, outputs, rewards):
        trainer.step([query], [response], torch.tensor([reward]))


if __name__ == "__main__":
    test_grpo = GRPO(model1="Qwen/Qwen2.5-7B-Instruct", 
                     model2="Qwen/Qwen2.5-7B-Instruct", 
                     base_url="http://127.0.0.1:8000")
    
    test_item = load_product(dataset_dir, product_index=5)

    #Commented out until PPO functiality is needed
    #tokenizer = AutoTokenizer.from_pretrained(test_grpo.model1)
    #model = AutoModelForCausalLM.from_pretrained(test_grpo.model1)

    #test_ppo_trainer = PPOTrainer(test_grpo.config, model, tokenizer)

    run_single_grpo(GRPO=test_grpo, item=test_item, max_turns=10)
