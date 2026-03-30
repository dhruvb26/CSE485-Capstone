import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict
import subprocess
import time
from datetime import datetime
import requests

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

from main import (
    _get_client,
    run_dialog,   
)

DATASET_PATH = '/home/bbreisc1/cse485-capstone/data/amazon_history_test'
ITEM_LIMIT = 20
F = 0.8
MAX_TURNS = 12

MISTRAL_INSTRUCT = {'name': 'mistral_instruct', 
                    'path': '/scratch/bbreisc1/bbreisc1/models/mistralai__Mistral-7B-Instruct-v0.3', 
                    'cache_dir': '/scratch/bbreisc1/bbreisc1/cahce/mistral_instruct'}
QWEN_CHAT = {'name': 'qwen_chat', 
                    'path': '/scratch/bbreisc1/bbreisc1/models/Qwen__Qwen-7B-Chat', 
                    'cache_dir': '/scratch/bbreisc1/bbreisc1/cahce/qwen_chat'}
QWEN_INSTRUCT = {'name': 'qwen_instruct', 
                    'path': '/scratch/bbreisc1/bbreisc1/models/qwen_model', 
                    'cache_dir': '/scratch/bbreisc1/bbreisc1/cahce/qwen_instruct'}
QWEN_INSTRUCT_RL = {'name': 'qwen_instruct_rl', 
                    'path': '/scratch/bbreisc1/bbreisc1/grpo_qwen_checkpoint247', 
                    'cache_dir': '/scratch/bbreisc1/bbreisc1/cahce/qwen_instruct_rl'}
LLAMA_CHAT = {'name': 'llama_chat', 
                    'path': '/scratch/bbreisc1/bbreisc1/models/meta-llama__Llama-2-13b-chat-hf', 
                    'cache_dir': '/scratch/bbreisc1/bbreisc1/cache/llama_chat'}

RUNS = [
    (MISTRAL_INSTRUCT, QWEN_INSTRUCT_RL),
    (LLAMA_CHAT, QWEN_INSTRUCT_RL),
    (MISTRAL_INSTRUCT, QWEN_INSTRUCT),
    (LLAMA_CHAT, QWEN_INSTRUCT)
    ]

def debug_tokenize(base_url, model_path, messages):
    r = requests.post(
        f"{base_url}/tokenize",
        headers={"Content-Type": "application/json"},
        json={
            "model": model_path,
            "messages": messages,
            "add_special_tokens": True
        }
    )
    data = r.json()
    # decode the tokens back to text so you can read it
    r2 = requests.post(
        f"{base_url}/detokenize",
        headers={"Content-Type": "application/json"},
        json={
            "model": model_path,
            "tokens": data["tokens"]
        }
    )
    print(r2.json().get("prompt", "NO PROMPT"))

def run_session(buyer_model: dict, seller_model: dict, product_limit: int, dataset_dir: str) -> Dict:
    SNPB = 0.00
    SNPS = 0.00
    DEALS = 0
    RESTART_EVERY = 40

    def start_servers():
        buyer_process = start_vllm_wait(model_path=buyer_model['path'], port=8000, cache_dir=buyer_model['cache_dir'], gpu_id=0)
        seller_process = start_vllm_wait(model_path=seller_model['path'], port=8003, cache_dir=seller_model['cache_dir'], gpu_id=1)
        buyer_client = _get_client(buyer_model['path'], 'http://127.0.0.1:8000')
        seller_client = _get_client(seller_model['path'], 'http://127.0.0.1:8003')
        return buyer_process, seller_process, buyer_client, seller_client

    def restart_servers(buyer_process, seller_process):
        print("Restarting vLLM servers...")
        stop_vllm(buyer_process)
        stop_vllm(seller_process)
        subprocess.run("pkill -f vllm", shell=True)
        time.sleep(5)
        return start_servers()

    try:
        buyer_process, seller_process, buyer_client, seller_client = start_servers()

        debug_tokenize(
            "http://127.0.0.1:8000",
            buyer_model['path'],
            [
                {"role": "system", "content": "test system prompt"},
                {"role": "user", "content": "test user message"},
            ]
        )
    except Exception as e:
        print(f'Error: error starting vllm servers: {e}')
        subprocess.run("pkill -f vllm", shell=True)
        time.sleep(3)
        return []

    history = []
    for i in range(product_limit):

        # Periodic restart
        if i > 0 and i % RESTART_EVERY == 0:
            print(f"\n[{i}/{product_limit}] Scheduled restart...")
            try:
                buyer_process, seller_process, buyer_client, seller_client = restart_servers(buyer_process, seller_process)
                print("Servers restarted successfully.\n")
            except Exception as e:
                print(f"Failed to restart servers: {e}")
                subprocess.run("pkill -f vllm", shell=True)
                break

        try:
            item = load_product(dataset_dir, product_index=i)
        except IndexError:
            break

        B = F * float(item["highest_price"])
        C = float(item["lowest_price"])
        inv_block = inventory_list(item)
        shop_block = shopping_list(item, B)
        buyer = BuyerAgent(
            client=buyer_client,
            model_name=buyer_model['name'],
            inv_block=inv_block,
            shop_block=shop_block,
            B=B,
            code=item["codename"],
            max_turns=MAX_TURNS,
        )
        seller = SellerAgent(
            client=seller_client,
            model_name=seller_model['name'],
            inv_block=inv_block,
            C=C,
            code=item["codename"],
            max_turns=MAX_TURNS,
        )
        res = run_dialog(
            buyer=buyer,
            seller=seller,
            item=item,
            B=B,
            C=C,
            max_turns=MAX_TURNS,
            log_file=None,
        )
        outcome = res[0]
        dialog = res[1]
        #print(json.dumps(dialog, indent=4, default=lambda o: o.__dict__))
        m = compute_metrics(outcome, B, C)
        run = {
            'buyerPath': buyer_model['path'],
            'buyerName': buyer_model['name'],
            'sellerPath': seller_model['path'],
            'sellerName': seller_model['name'],
            'item': item,
            'b': B,
            'c': C,
            'outcome': outcome,
            'metrics': m
        }
        history.append(run)

        if outcome['deal']:
            DEALS += 1
        SNPB += m['NPb']
        SNPS += m['NPs']

        print(f"[{i+1}/{product_limit}] Deal: {outcome['deal']} | Price: {outcome.get('price', 'N/A')} | DEALS so far: {DEALS}")

    stop_vllm(buyer_process)
    stop_vllm(seller_process)
    session = {
        'buyer': buyer_model['path'],
        'seller': seller_model['path'],
        'deals': DEALS,
        'deal_rate': DEALS / max(len(history), 1),
        'SNPB': SNPB,
        'SNPS': SNPS,
        'history': history
    }
    return session


if __name__ == "__main__":
    sessions = []
    for pair in RUNS:
        subprocess.run("pkill -f vllm", shell=True)
        time.sleep(5)

        session = run_session(buyer_model=pair[0], seller_model=pair[1], product_limit=279, dataset_dir=DATASET_PATH)
        sessions.append(session)
        print(f'-------------Buyer: {session["buyer"]}\nSeller: {session["seller"]}\nDeal Rate: {session["deal_rate"]}\nSNPB: {session["SNPB"]}\nSNPS: {session["SNPS"]}----------')
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"results_{timestamp}_{pair[0]['name']}_{pair[1]['name']}.json"
        with open(filename, "w") as f:
            json.dump(session, f, indent=4, default=lambda o: o.__dict__)