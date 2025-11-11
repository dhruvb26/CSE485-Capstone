import argparse
import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

#GLOBAL VARS
CRAIGSLIST_PATH_RELATIVE = "../data/craigslist_bargains/train.json"
FORMATTED_PATH_RELATIVE = "formatted_data/craigslist_formatted.json"

#Instruction prompts were designed to vary from the evaluation format and not include defined actions but instead natural dialogue
BUYER_INSTRUCTION = "You are a buyer looking to buy an item from craigslist, {item_title}, and want to get it for the best price you can while still making an effort to agree on a deal. The description of the item is stated as: {description}. The item is priced at {item_price}, and your target price is {buyer_target}. Your task as the buyer is to negotiating with the seller and reach a deal. Negotiate naturally with the seller based on the previous dialogue."
SELLER_INSTRUCTION = "You are a seller looking to sell an item on craigslist, {item_title}, and want to sell it for the best price you can while still making an effort to agree on a deal. The description of the item is stated as: {description}. You priced the item at {item_price}, and your target price is {seller_target}. Your task as the seller is to negotiating with the buyer and reach a deal. Negotiate naturally with the buyer based on the previous dialogue."

def parse_craigslist_bargains(raw_dataset_path: str, output_dataset_path, buyer_instruction , seller_instruction, allow_incomplete_bargains: bool):
    training_instances = []
    
    with open(raw_dataset_path) as f:
        dialogues = json.load(f)

        #print(json.dumps(dialogues[0], indent=4))
        #Buyer always messages first
        for dialogue in dialogues:
            scenario = dialogue["scenario"]
            category = scenario["category"]
            buyer_kb = scenario["kbs"][0]
            seller_kb = scenario["kbs"][1]
            buyer_target = buyer_kb["personal"]["Target"]
            seller_target = seller_kb["personal"]["Target"]
            item = seller_kb["item"]
            price = item["Price"]
            title = item["Title"]
            description = item["Description"]
            messages = dialogue["events"]
            reward = dialogue["outcome"]["reward"]

            message_history = ""

            #check to allow incomplete bargains
            if not allow_incomplete_bargains and reward == 0:
                break

            buyer_inst = BUYER_INSTRUCTION.format(
                item_title = title,
                description = description,
                item_price = price,
                buyer_target = buyer_target
            )
            seller_inst = SELLER_INSTRUCTION.format(
                item_title = title,
                description = description,
                item_price = price,
                seller_target = seller_target
            )  
            
            for i, message in enumerate(messages):
                training_instance = {}
                next_message_role = message["agent"]
                
                #check if message is last message in sequence
                if i == len(messages) - 1:
                    break

                #fromat output
                output = ""
                if message["action"] == "message":
                    output = message["data"]
                elif message["action"] == "offer":
                    output = f'I offer {message["data"]["price"]} for it'
                elif message["action"] == "accept":
                    output = "I accept the offer"
                else:
                    output = "I reject the offer"


                #check who sends next message, buyer: next_message_role = 0, seller: next_message_role = 1
                if next_message_role:
                    training_instance["instruction"] = seller_inst
                    training_instance["input"] = message_history
                    training_instance["output"] = output
                    message_history = message_history + f'Seller: {output}\n'
                else:
                    training_instance["instruction"] = buyer_inst
                    training_instance["input"] = message_history
                    training_instance["output"] = output
                    message_history = message_history + f'Buyer: {output}\n'

                training_instances.append(training_instance)
    
    with open(output_dataset_path, "w") as f:
        json.dump(training_instances, f, indent=4)
    
    return training_instances
                

if __name__ == "__main__":
    print("Processing raw craigslist/bargains")
    instances = parse_craigslist_bargains(CRAIGSLIST_PATH_RELATIVE, FORMATTED_PATH_RELATIVE, BUYER_INSTRUCTION, SELLER_INSTRUCTION, True)
    print(f'Number of training tuples: {len(instances)}')
    print(json.dumps(instances[0], indent=4))
    print(json.dumps(instances[1], indent=4))



