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
        self.config = PPOConfig(
            model_name=model1,
            learning_rate=1.41e-5,
            batch_size=1,
            log_with=None)

    def add_run(self):
        self.run_incrementer += 1

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
        outcome = run_dialog(
            buyer=buyer_copy,
            seller=seller_copy,
            item=item,
            B=B,
            C=C,
            max_turns=max_turns,
            log_file=session_log_file,
            temperature= 0.7,
            top_p = 0.9)
        outcomes[i] = outcome
        m = compute_metrics(outcome, B, C)
        metrics[i] = m
    print(outcomes)
    print(metrics)

    return outcomes, metrics

def run_single_ppo(trainer, inputs, outputs, rewards):
    for query, response, reward in zip(inputs, outputs, rewards):
        trainer.step([query], [response], torch.tensor([reward]))


if __name__ == "__main__":
    test_grpo = GRPO(model1="mistralai/Mistral-7B-Instruct-v0.2", 
                     model2="mistralai/Mistral-7B-Instruct-v0.2", 
                     base_url="http://127.0.0.1:8000")
    
    test_item = load_product(dataset_dir, product_index=5)

    tokenizer = AutoTokenizer.from_pretrained(test_grpo.model1)
    model = AutoModelForCausalLM.from_pretrained(test_grpo.model1)

    test_ppo_trainer = PPOTrainer(test_grpo.config, model, tokenizer)

    run_single_grpo(GRPO=test_grpo, item=test_item, max_turns=10)
