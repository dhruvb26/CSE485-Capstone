import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import logging
from datetime import datetime
from pathlib import Path
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"grpo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logger = logging.getLogger("GRPO")
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

from typing import Dict
import time
import copy
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import bitsandbytes as bnb
import gc


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
    start_vllm_wait,
    stop_vllm
)
from main import _get_client
from main import run_dialog


'''
This GRPO Layout is to follow a simple terminal reward self play RL Design, this is an initial design and code implementation
'''

dataset_dir = 'data/amazon_history_price' #directory of location of dataset
product_limit = 10 #limit on number of products to train on 

class GRPO:


    def __init__(self, model1, model2, base_url):
        self.model1 = _get_client(model_name=model1, local_base_url=base_url)
        logger.info(f'Model1 Loaded: {model1}')
        self.model1_name = model1
        self.model2 = _get_client(model_name=model2, local_base_url=base_url)
        logger.info(f'Model2 Loaded: {model2}')
        self.model2_name = model2
        self.runs = {}
        self.run_incrementer = 0
        self.model1_tokenizer = AutoTokenizer.from_pretrained("/scratch/bbreisc1/bbreisc1/qwen_model", local_files_only=True)
        self.model2_tokenizer = AutoTokenizer.from_pretrained("/scratch/bbreisc1/bbreisc1/qwen_model", local_files_only=True)
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
    for i in range(start, len(negotiation_log), 2):
        input_tokens, temp = chat_http_package_to_model_input(tokenizer=tokenizer, http_package=negotiation_log[i][0])
        output_tokens, temp = chat_http_package_from_model_output(tokenizer=tokenizer, http_package=negotiation_log[i][1])
        queries.append(input_tokens)
        responses.append(output_tokens)
        advantages.append(advantage)

    return queries, responses, advantages


def compute_grpo_loss(model, ref_model, query_tokens, response_tokens, advantage,
                      epsilon=0.2, beta=0.02):
    device = next(model.parameters()).device  # get whatever device model is on
    
    input_ids = torch.cat([query_tokens, response_tokens]).unsqueeze(0).to(device)
    response_tokens = response_tokens.to(device)
    resp_len = response_tokens.size(0)

    with torch.no_grad():
        ref_logits = ref_model(input_ids).logits[0, -resp_len-1:-1]
        ref_logprobs = F.log_softmax(ref_logits, dim=-1)
        ref_token_logprobs = ref_logprobs.gather(
            -1, response_tokens.unsqueeze(-1)).squeeze(-1)

    logits = model(input_ids).logits[0, -resp_len-1:-1]
    logprobs = F.log_softmax(logits, dim=-1)
    token_logprobs = logprobs.gather(-1, response_tokens.unsqueeze(-1)).squeeze(-1)

    old_logprob = ref_token_logprobs.sum()
    new_logprob = token_logprobs.sum()
    ratio = torch.exp(new_logprob - old_logprob)

    clip_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    policy_loss = -torch.min(ratio * advantage, clip_ratio * advantage)

    kl = (ref_token_logprobs.exp() * (ref_token_logprobs - token_logprobs)).sum()

    return policy_loss + beta * kl


def grpo_update_offline(model, ref_model, optimizer,
                        queries, responses, advantages,
                        batch_size=8, epsilon=0.2, beta=0.02):
    adv = torch.tensor(advantages, dtype=torch.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    model.train()
    optimizer.zero_grad()
    total_loss = 0.0  # change 3 — plain float, not a tensor

    for i, (q, r, a) in enumerate(zip(queries, responses, adv.tolist())):
        q_t = torch.tensor(q, dtype=torch.long)
        r_t = torch.tensor(r, dtype=torch.long)
        loss = compute_grpo_loss(model, ref_model, q_t, r_t, a, epsilon, beta)
        
        loss.backward()  # change 3 — backward immediately, frees graph
        total_loss += loss.item()  # just a float for logging

        if (i + 1) % batch_size == 0 or (i + 1) == len(queries):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss = 0.0

def load_models_for_training(weights_path):
    model = AutoModelForCausalLM.from_pretrained(
        weights_path, torch_dtype=torch.bfloat16, device_map="cuda:1")
    ref_model = AutoModelForCausalLM.from_pretrained(
        weights_path, torch_dtype=torch.bfloat16, device_map="cuda:1")
    ref_model.eval()
    ref_model.requires_grad_(False)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-6)
    return model, ref_model, optimizer

def free_models(model, ref_model, optimizer=None):
    model.cpu()
    ref_model.cpu()
    del model, ref_model
    if optimizer is not None:
        del optimizer
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.ipc_collect()



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
    logger.info('GRPO Class Created')
    
    #test_item = load_product(dataset_dir, product_index=5)

    #Commented out until PPO functiality is needed
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained("/scratch/bbreisc1/bbreisc1/qwen_model")
    logger.info('Model loaded')


    #updated/session configurations !!! Need to update save_path at least
    #URGENT!
    update_every = 64
    load_every = 2
    product_limit = 6
    save_path = f"/scratch/bbreisc1/bbreisc1/grpo_qwen_checkpoint"
    cache_directory= f'/scratch/bbreisc1/bbreisc1/hf_cache_qwen/'
    current_weights_path = "Qwen/Qwen2.5-7B-Instruct"
    to_load_from_path = "/scratch/bbreisc1/bbreisc1/qwen_model"
    #not a parameter
    update_count = 0
    checkpoint_count = 0

    #History Buffers
    buyer_buffer_q = []
    buyer_buffer_r = []
    buyer_buffer_a = []

    seller_buffer_q = []
    seller_buffer_r = []
    seller_buffer_a = []

    #start vllm server with OG weights
    logger.info(f'Starting vLLM server with OG weights')
    proc = start_vllm_wait("Qwen/Qwen2.5-7B-Instruct", cache_dir=cache_directory)
    logger.info(f'vLLM started with PID {proc.pid}')

    try:
        for i in range(product_limit):
            item = load_product(dataset_dir, product_index=i)
            #log item loaded
            logger.info(f"Item loaded: {item['title']}")

            result = run_single_grpo(
                GRPO=test_grpo,
                item=item,
                max_turns=10
            )
            #log session ran
            logger.info(f"Session complete for item: {item['title']}")

            buyer_buffer_q.extend(result["buyer_queries"])
            buyer_buffer_r.extend(result["buyer_responses"])
            buyer_buffer_a.extend(result["buyer_advantage"])

            seller_buffer_q.extend(result["seller_queries"])
            seller_buffer_r.extend(result["seller_responses"])
            seller_buffer_a.extend(result["seller_advantage"])
            #log results extracted
            logger.info(f"Results extracted for session over {item['title']}")
            #maybe log results

            if len(buyer_buffer_q) + len(seller_buffer_q) >= update_every:
                stop_vllm(proc)
                time.sleep(15)  # increase from 10 to 15
                gc.collect()
                torch.cuda.empty_cache()
                logger.info(f"GPU memory allocated before loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
                logger.info(f"GPU memory reserved before loading: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")
                
                
                #log updated starting
                logger.info(f'Starting weight update #{update_count + 1}, with #{len(buyer_buffer_q) + len(seller_buffer_q)} updates')


                model, ref_model, optimizer = load_models_for_training(current_weights_path)
                model.gradient_checkpointing_enable()

                grpo_update_offline(model, ref_model, optimizer,
                    buyer_buffer_q, buyer_buffer_r, buyer_buffer_a)
                #log buyer updated completed
                logger.info(f'Buyer updates complete, #{len(buyer_buffer_q)} updateds completed')

                buyer_buffer_q.clear()
                buyer_buffer_r.clear()
                buyer_buffer_a.clear()
                #log buyer buffers cleared
                logger.info(f'Buyer buffer cleared')

                grpo_update_offline(model, ref_model, optimizer,
                    seller_buffer_q, seller_buffer_r, seller_buffer_a)
                #log seller updates complete
                logger.info(f'Seller updates complete, #{len(seller_buffer_q)} updateds completed')

                seller_buffer_q.clear()
                seller_buffer_r.clear()
                seller_buffer_a.clear()
                #log seller buffers cleared
                logger.info(f'Seller buffer cleared')

                update_count += 1
                #log update count
                logger.info(f'Finished weight updated #{update_count}')

                model.save_pretrained(f'{save_path}{update_count}')
                tokenizer.save_pretrained(f'{save_path}{update_count}')
                to_load_from_path = f'{save_path}{update_count}'
                free_models(model, ref_model, optimizer)

                if update_count % load_every == 0:
                    current_weights_path = f'{save_path}{update_count}'
                    logger.info(f'Loading updated weights into vLLM!')

                time.sleep(2)
                logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
                logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")
                proc = start_vllm_wait(model_path=current_weights_path, cache_dir=cache_directory)
                #log vllm server restarted
                logger.info(f'Updated checkpoint vLLM server started, PID: {proc.pid}')

    finally:
        stop_vllm(proc)