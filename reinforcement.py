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

#Helper function used to convert vllm (chat) style payload to actual model prompt, takes input package as http_package
def chat_http_package_to_model_input(tokenizer, http_package):
    #extract messages field
    messages = http_package['messages']

    #format prompt from messages, tokenize
    prompt_tokens = tokenizer.apply_chat_template(messages, tokenize=True)

    # convert back to verify
    decoded = tokenizer.decode(prompt_tokens)

    return (prompt_tokens, decoded)

#Helper function used to convert vllm (chat) style payload to actual model output, takes output package as http_package
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


#This function takes in a full negotiation log as well as the output metrics, and aims to return 3 lists for the buyer agents, the reward for each input->output pair is the same as the terminal reward
def extract_input_output_tadvanatge_list(tokenizer, negotiation_log, reward_metric, reward_average, seller: bool) -> tuple:
    queries = []
    responses = []
    advantages = []
    reward = reward_metric["NPb"]
    advantage = reward - reward_average
    start = 0
    if seller:
        start = 1
    for i in range(start=start, stop=len(negotiation_log), step=2):
        input_tokens, temp = chat_http_package_to_model_input(tokenizer=tokenizer, http_package=negotiation_log[i][0])
        output_tokens, temp = chat_http_package_from_model_output(tokenizer=tokenizer, http_package=negotiation_log[i][1])
        queries.append(input_tokens)
        responses.append(output_tokens)
        advantages.append(advantage)

    return queries, responses, advantages


def grpo_update(ppo_trainer, queries, responses, advantages):
    # Normalize advantages (VERY important)
    adv = torch.tensor(advantages)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    for query, response, advantage in zip(queries, responses, adv):
        query_tensor = torch.tensor(query, dtype=torch.long)
        response_tensor = torch.tensor(response, dtype=torch.long)
        reward_tensor = torch.tensor([advantage.item()])

        # PPOTrainer.step handles:
        # - old logprobs
        # - ratio clipping
        # - KL penalty
        ppo_trainer.step(
            [query_tensor],
            [response_tensor],
            reward_tensor)
        
def grpo_update_batch(ppo_trainer, queries, responses, advantages):
    adv = torch.tensor(advantages)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    query_tensors = [torch.tensor(q, dtype=torch.long) for q in queries]
    response_tensors = [torch.tensor(r, dtype=torch.long) for r in responses]
    reward_tensors = adv.tolist()

    ppo_trainer.step(query_tensors, response_tensors, reward_tensors)


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
    #runs includes the negotiation log, outcome, and metrics for each negotiation
    runs = []
    #tracks the sum of buyer reward over all sessions
    buyer_reward_sum = 0.00
    #tracks the sum of seller reward over all sessions
    seller_reward_sum = 0.00
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
        runs.append({'negotiation_log': negotiation_log, 'outcome': outcome, 'metric': m})
        seller_reward_sum += m['NPs']
        buyer_reward_sum += m['NPb']
    buyer_reward_average = buyer_reward_sum / 8
    seller_reward_average = seller_reward_sum / 8
    print(metrics)
    #buyer_log_encoded, seller_log_encoded, buyer_log_decoded, seller_log_decoded = extract_turns_from_all_messages(tokenizer_buyer=GRPO.model1_tokenizer, tokenizer_seller=GRPO.model2_tokenizer, negotiation_log=negotiation_log["allPackagesSentOrdered"])
    buyer_queries = []
    buyer_responses = []
    buyer_advantage = []
    seller_queries = []
    seller_responses = []
    seller_advantage = []
    for session in runs:
        bq, br, ba = extract_input_output_tadvanatge_list(tokenizer=GRPO.model1_tokenizer, 
                                                          negotiation_log=session['negotiation_log']['allPackagesSentOrdered'], 
                                                          reward_metric=session['metric'], 
                                                          reward_average=buyer_reward_average, 
                                                          seller=False)
        buyer_queries.extend(bq)
        buyer_responses.extend(br)
        buyer_advantage.extend(ba)
        sq, sr, sa = extract_input_output_tadvanatge_list(tokenizer=GRPO.model2_tokenizer, 
                                                          negotiation_log=session['negotiation_log']['allPackagesSentOrdered'], 
                                                          reward_metric=session['metric'], 
                                                          reward_average=seller_reward_average, 
                                                          seller=True)
        seller_queries.extend(sq)
        seller_responses.extend(sr)
        seller_advantage.extend(sa)
        
    return {"outcomes": outcomes,
        "metrics": metrics,
        "buyer_queries": buyer_queries,
        "buyer_responses": buyer_responses,
        "buyer_advantage": buyer_advantage,
        "seller_queries": seller_queries,
        "seller_responses": seller_responses,
        "seller_advantage": seller_advantage}

if __name__ == "__main__":
    test_grpo = GRPO(model1="Qwen/Qwen2.5-7B-Instruct", 
                     model2="Qwen/Qwen2.5-7B-Instruct", 
                     base_url="http://127.0.0.1:8000")
    
    #test_item = load_product(dataset_dir, product_index=5)

    #Commented out until PPO functiality is needed
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    config = PPOConfig(
        learning_rate=1e-6,
        batch_size=4,
        mini_batch_size=1,
        kl_coef=0.02,        # important for stability
    )

    ppo_trainer = PPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
    )

    #updated/session configurations !!! Need to update save_path at least
    #URGENT!
    update_every = 64
    save_every = 5
    product_limit = 50
    save_path = f"/scratch/bbreisc1/bbreisc1/grpo_qwen_checkpoint"
    #not a parameter
    update_count = 0

    #History Buffers
    buyer_buffer_q = []
    buyer_buffer_r = []
    buyer_buffer_a = []

    seller_buffer_q = []
    seller_buffer_r = []
    seller_buffer_a = []

    for i in range(product_limit):
        item = load_product(dataset_dir, product_index=i)
        #log item loaded

        result = run_single_grpo(
            GRPO=test_grpo,
            item=item,
            max_turns=10
        )
        #log session ran

        buyer_buffer_q.extend(result["buyer_queries"])
        buyer_buffer_r.extend(result["buyer_responses"])
        buyer_buffer_a.extend(result["buyer_advantage"])

        seller_buffer_q.extend(result["seller_queries"])
        seller_buffer_r.extend(result["seller_responses"])
        seller_buffer_a.extend(result["seller_advantage"])
        #log results extracted
        #maybe log results

        if len(buyer_buffer_q) + len(seller_buffer_q) >= update_every:
            #log updated starting

            grpo_update_batch(
                ppo_trainer,
                buyer_buffer_q,
                buyer_buffer_r,
                buyer_buffer_a
            )
            #log buyer updated completed

            buyer_buffer_q.clear()
            buyer_buffer_r.clear()
            buyer_buffer_a.clear()
            #log buyer buffers cleared

            grpo_update_batch(
                ppo_trainer,
                seller_buffer_q,
                seller_buffer_r,
                seller_buffer_a
            )
            #log seller updated completed

            seller_buffer_q.clear()
            seller_buffer_r.clear()
            seller_buffer_a.clear()
            #log seller buffers cleared

            update_count += 1
            #log update count


            if update_count % save_every == 0:
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                #log checkpoint saved

                #restart vllm server with new model weights