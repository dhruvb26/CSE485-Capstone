# Eval Tasks

The paper [Are LLMs Effective Negotiators? Systematic Evaluation of the
Multifaceted Capabilities of LLMs in Negotiation Dialogues](https://arxiv.org/pdf/2402.13550) and their [repository](https://github.com/DSincerity/SysEval-NegoLLMs) outlines about 35 tasks across the 4 datasets. These tasks are organized into 3 stages: START, DURING, and END to simulate the negotiation conversation and test every LLM's performance across scenarios. To keep things simple for now we're going to go forward only dealing with DND (Deal or No Deal) and CA (CaSiNo) datasets. 

These were mainly chosen based on the availability of data, ease of use and formatting, also how well they can match up with the original paper that introduced the Amazon Price History dataset. We've chosen objective tasks out of the selected datasetes. Here's a complete list of our chosen tasks: 

## CA (CaSiNo) 

1. **[sta_total_item_count_ca](https://github.com/DSincerity/SysEval-NegoLLMs/blob/main/tasks/sta_total_item_count_ca.py)** — Count the total number of items being negotiated over (no dialogue; uses item counts and self point values).

2. **[sta_max_points_ca](https://github.com/DSincerity/SysEval-NegoLLMs/blob/main/tasks/sta_max_points_ca.py)** — Determine the maximum possible points you can get in any deal (no dialogue).

3. **[sta_ask_point_values_ca](https://github.com/DSincerity/SysEval-NegoLLMs/blob/main/tasks/sta_ask_point_values_ca.py)** — Report how many points one package of each issue (food, water, firewood) is worth to you, as a dict (no dialogue).

4. **[sta_ask_high_priority_ca](https://github.com/DSincerity/SysEval-NegoLLMs/blob/main/tasks/sta_ask_high_priority_ca.py)** — Identify your highest-priority issue among food, water, and firewood (no dialogue).

5. **[sta_ask_low_priority_ca](https://github.com/DSincerity/SysEval-NegoLLMs/blob/main/tasks/sta_ask_low_priority_ca.py)** — Identify your lowest-priority issue among food, water, and firewood (no dialogue).

6. **[mid_strategy_ca](https://github.com/DSincerity/SysEval-NegoLLMs/blob/main/tasks/mid_strategy_ca.py)** — Classify which negotiation strategies appear in a given utterance (e.g. small-talk, empathy, coordination, no-need, elicit-pref, etc.), using dialogue context.

7. **[mid_ask_high_priority_ca](https://github.com/DSincerity/SysEval-NegoLLMs/blob/main/tasks/mid_ask_high_priority_ca.py)** — Same as sta high priority but with partial dialogue history.

8. **[mid_ask_low_priority_ca](https://github.com/DSincerity/SysEval-NegoLLMs/blob/main/tasks/mid_ask_low_priority_ca.py)** — Same as sta low priority but with partial dialogue history.

9. **[mid_partner_ask_high_priority_ca](https://github.com/DSincerity/SysEval-NegoLLMs/blob/main/tasks/mid_partner_ask_high_priority_ca.py)** — Infer your partner's highest-priority issue from partial dialogue.

10. **[mid_partner_ask_low_priority_ca](https://github.com/DSincerity/SysEval-NegoLLMs/blob/main/tasks/mid_partner_ask_low_priority_ca.py)** — Infer your partner's lowest-priority issue from partial dialogue.

11. **[end_deal_specifics_ca](https://github.com/DSincerity/SysEval-NegoLLMs/blob/main/tasks/end_deal_specifics_ca.py)** — After the full dialogue, report how many items you got for each issue (food, water, firewood) in the final deal; 0 if no deal.

12. **[end_deal_total_ca](https://github.com/DSincerity/SysEval-NegoLLMs/blob/main/tasks/end_deal_total_ca.py)** — After the full dialogue, report how many total points you got at the end of the negotiation.


## DND (Deal or No Deal)

1. **[sta_total_item_count_dnd](https://github.com/DSincerity/SysEval-NegoLLMs/blob/main/tasks/sta_total_item_count_dnd.py)** — Count the total number of items on the table (no dialogue).

2. **[sta_max_points_dnd](https://github.com/DSincerity/SysEval-NegoLLMs/blob/main/tasks/sta_max_points_dnd.py)** — Determine the maximum number of points you can possibly get in any deal (no dialogue).

3. **[sta_ask_point_values_dnd](https://github.com/DSincerity/SysEval-NegoLLMs/blob/main/tasks/sta_ask_point_values_dnd.py)** — Report how many points one item of each issue (books, hats, balls) is worth to you, as a dict (no dialogue).

4. **[mid_dial_act_dnd](https://github.com/DSincerity/SysEval-NegoLLMs/blob/main/tasks/mid_dial_act_dnd.py)** — Classify the dialogue act of an utterance (e.g. greet, inquire, propose, agree, disagree, insist, unknown), with context.

5. **[mid_full_proposal_dnd](https://github.com/DSincerity/SysEval-NegoLLMs/blob/main/tasks/mid_full_proposal_dnd.py)** — From an utterance, extract the full proposal: how many books, hats, and balls the speaker gets (dict).

6. **[end_deal_specifics_dnd](https://github.com/DSincerity/SysEval-NegoLLMs/blob/main/tasks/end_deal_specifics_dnd.py)** — After the full dialogue, report how many items you got for each issue (books, hats, balls) in the final deal; 0 if no deal.

7. **[end_deal_total_dnd](https://github.com/DSincerity/SysEval-NegoLLMs/blob/main/tasks/end_deal_total_dnd.py)** — After the full dialogue, report how many total points you got; 0 if no deal.

## How to Run

To run the tasks, you can use the following command:

### Using Hugging Face Models

```bash
# Run DND tasks with Mistral-7B
python main.py --storage_dir storage/ --dataset_name dnd --model_name hf_model --hf_model_str "mistralai/Mistral-7B-Instruct-v0.1" --task_name "sta_total_item_count_dnd,sta_max_points_dnd,sta_ask_point_values_dnd,mid_dial_act_dnd,mid_full_proposal_dnd,end_deal_specifics_dnd,end_deal_total_dnd" --num_instances 200

# Run CA tasks with Mistral-7B
 python main.py --storage_dir storage/ --dataset_name casino --model_name hf_model --hf_model_str "mistralai/Mistral-7B-Instruct-v0.1" --task_name "sta_total_item_count_ca,sta_max_points_ca,sta_ask_point_values_ca,sta_ask_high_priority_ca,sta_ask_low_priority_ca,mid_strategy_ca,mid_ask_high_priority_ca,mid_ask_low_priority_ca,mid_partner_ask_high_priority_ca,mid_partner_ask_low_priority_ca,end_deal_specifics_ca,end_deal_total_ca" --num_instances 200
```

### Using OpenAI Models

```bash
# Run DND tasks with GPT-4o-mini
python main.py --storage_dir storage/ --dataset_name dnd --model_name open_ai --openai_model_str "gpt-4o-mini-2024-07-18" --task_name "sta_total_item_count_dnd,sta_max_points_dnd,sta_ask_point_values_dnd,mid_dial_act_dnd,mid_full_proposal_dnd,end_deal_specifics_dnd,end_deal_total_dnd" --num_instances 200

# Run CA tasks with GPT-4o-mini
python main.py --storage_dir storage/ --dataset_name casino --model_name open_ai --openai_model_str "gpt-4o-mini-2024-07-18" --task_name "sta_total_item_count_ca,sta_max_points_ca,sta_ask_point_values_ca,sta_ask_high_priority_ca,sta_ask_low_priority_ca,mid_strategy_ca,mid_ask_high_priority_ca,mid_ask_low_priority_ca,mid_partner_ask_high_priority_ca,mid_partner_ask_low_priority_ca,end_deal_specifics_ca,end_deal_total_ca" --num_instances 200
```

The commands above can be run in the fork of the original repo [here](https://github.com/dhruvb26/SysEval-NegoLLMs) because we've added the ability to run the tasks with both Hugging Face and OpenAI models in newer versions of the libraries. 
