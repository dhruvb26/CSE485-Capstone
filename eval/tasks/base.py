"""
Base classes for all task handlers.

Merged from SysEval-NegoLLMs k_task.py and task_w.py.
Both provide prompt construction, prediction extraction, logging, and deduplication.
"""

import copy
import json

from eval import utils


def _extract_last_answer(text):
    """Return the content of the last <answer>...</answer> block, or None."""
    end = text.rfind("</answer>")
    if end == -1:
        return None
    start = text.rfind("<answer>", 0, end)
    if start == -1:
        return None
    return text[start + len("<answer>"):end].strip()


def _extract_last_json_answer(raw):
    """Return the content of the last <answer>{...}</answer> block, or None."""
    search_end = len(raw)
    while True:
        end = raw.rfind("</answer>", 0, search_end)
        if end == -1:
            return None
        start = raw.rfind("<answer>", 0, end)
        if start == -1:
            return None
        inner = raw[start + len("<answer>"):end].strip()
        if inner.startswith("{") and inner.endswith("}"):
            return inner
        search_end = start


class KBaseTaskHandler:
    """Base handler for every task (K-variant)."""

    def __init__(self, name, args):
        self.name = name
        self.args = args

    def evaluate(self, dataset_handler, model_handler, return_prompt_gt=False):
        raise NotImplementedError

    def flatten(self, lst):
        new = []
        for item in lst:
            for element in item:
                new.append(element)
        return new

    def get_prompt_ca(self, instance, template, agent):
        dialogue = ""
        logs = instance['chat_logs']
        participant_info = instance['participant_info']

        history = logs[:]
        history2 = []
        for utt in history:
            if utt['text'] not in ["Submit-Deal", "Accept-Deal", "Walk-Away", "Reject-Deal"]:
                history2.append(utt)

        if self.args.num_utts_partial_dial != -1:
            history2 = history2[:self.args.num_utts_partial_dial]

        for utt in history2:
            dialogue += utt['id'] + ": " + utt['text'] + "\n"

        dialogue = dialogue.replace("mturk_agent_1:", "YOU:")
        dialogue = dialogue.replace("mturk_agent_2:", "THEM:")

        agent1_dict = participant_info["mturk_agent_1"]["value2issue"]
        agent2_dict = participant_info["mturk_agent_2"]["value2issue"]
        agent1_switched = {item: level for level, item in agent1_dict.items()}
        agent2_switched = {item: level for level, item in agent2_dict.items()}

        def priority2points(d):
            for k, v in d.items():
                if v == 'Low':
                    d[k] = 3
                elif v == 'Medium':
                    d[k] = 4
                else:
                    d[k] = 5
            return d

        agent1_points = priority2points(agent1_switched)
        agent2_points = priority2points(agent2_switched)

        prompt = template.replace("$dialogue$", dialogue)
        if agent == "mturk_agent_1":
            prompt = prompt.replace("$food_points$", str(agent1_points['Food']))
            prompt = prompt.replace("$water_points$", str(agent1_points['Water']))
            prompt = prompt.replace("$fire_points$", str(agent1_points['Firewood']))
        elif agent == "mturk_agent_2":
            prompt = prompt.replace("$food_points$", str(agent2_points['Food']))
            prompt = prompt.replace("$water_points$", str(agent2_points['Water']))
            prompt = prompt.replace("$fire_points$", str(agent2_points['Firewood']))

        return prompt

    def get_prompt_dnd(self, instance, template, agent):
        dialogue = ""
        dialogue_list = str(instance['dialogue']).split(" <eos> ")
        for turn in dialogue_list[:-1]:
            dialogue += turn + "\n"

        you_value = instance['input']["value"]
        them_value = instance['partner_input']["value"]
        counts = instance['input']['count']

        prompt = template.replace("$dialogue$", dialogue)
        prompt = prompt.replace("$num_books$", str(counts[0]))
        prompt = prompt.replace("$num_hats$", str(counts[1]))
        prompt = prompt.replace("$num_balls$", str(counts[2]))

        if agent == "YOU":
            prompt = prompt.replace("$book_points$", str(you_value[0]))
            prompt = prompt.replace("$hat_points$", str(you_value[1]))
            prompt = prompt.replace("$ball_points$", str(you_value[2]))
        elif agent == "THEM":
            prompt = prompt.replace("$book_points$", str(them_value[0]))
            prompt = prompt.replace("$hat_points$", str(them_value[1]))
            prompt = prompt.replace("$ball_points$", str(them_value[2]))

        return prompt

    def get_ul_prompt_dnd(self, instance, template):
        if "$dialogue$" in template:
            dialogue = ""
            for utterance_dict in instance:
                if utterance_dict["agent"] == 0:
                    dialogue += "YOU: " + utterance_dict["data"] + "\n"
                elif utterance_dict["agent"] == 1:
                    dialogue += "THEM: " + utterance_dict["data"] + "\n"
            prompt = template.replace("$dialogue$", dialogue)
        elif "$utterance$" in template:
            if instance["agent"] == 0:
                prompt = template.replace("$utterance$", "YOU: " + instance["data"].strip())
            elif instance["agent"] == 1:
                prompt = template.replace("$utterance$", "THEM: " + instance["data"].strip())
        return prompt

    def get_reg_slot_prompt_dnd(self, instance, template):
        base_prompt = self.get_ul_prompt_dnd(instance, template)
        book_count = instance["Counts"]["book"]
        hat_count = instance["Counts"]["hat"]
        ball_count = instance["Counts"]["ball"]
        prompt = base_prompt.replace("$num_books$", str(book_count)).replace("$num_hats$", str(hat_count)).replace("$num_balls$", str(ball_count))

        if "book_points" in prompt:
            if instance["agent"] == 0:
                book_value = instance["You_values"]["book"]
                hat_value = instance["You_values"]["hat"]
                ball_value = instance["You_values"]["ball"]
            else:
                book_value = instance["Them_values"]["book"]
                hat_value = instance["Them_values"]["hat"]
                ball_value = instance["Them_values"]["ball"]
            prompt = prompt.replace("$book_points$", str(book_value)).replace("$hat_points$", str(hat_value)).replace("$ball_points$", str(ball_value))

        return prompt

    def get_con_slot_prompt_dnd(self, instance, prev_instance, template):
        prompt = self.get_reg_slot_prompt_dnd(instance, template)
        if prev_instance:
            if prev_instance["agent"] == 0:
                prompt = prompt.replace("$previous_utterance$", "YOU: " + prev_instance["data"])
            elif prev_instance["agent"] == 1:
                prompt = prompt.replace("$previous_utterance$", "THEM: " + prev_instance["data"])
        else:
            prompt = prompt.replace("$previous_utterance$", "None")
        return prompt

    def get_reg_ul_prompt_cra(self, utterance, template):
        return template.replace("$utterance$", utterance)

    def get_con_ul_prompt_cra(self, utterance, context, template):
        prompt = template.replace("$utterance$", utterance)
        if isinstance(context, str):
            cxt = context.replace("\"", "").replace("\\n", " ").replace("\n", " ").split()
            cxt = [w for w in cxt if w != ""]
            cxt2 = []
            what = None
            for w in cxt:
                if w != "Alice:" and w != "Bob:":
                    cxt2.append(w)
                if w == "Alice:" or w == "Bob:":
                    if what is None or what != w:
                        cxt2.append(w)
                        what = w
            cxt2.reverse()
            cxt3 = []
            count = 0
            for w in cxt2:
                if w != "Alice:" and w != "Bob:":
                    cxt3.append(w)
                if w == "Alice:" or w == "Bob:":
                    cxt3.append(w)
                    count += 1
                if count == self.args.num_prior_utts:
                    break
            cxt3.reverse()
            cxt_str = " ".join(cxt3)
            prompt = prompt.replace("$previous_utterances$", cxt_str)
        else:
            prompt = prompt.replace("$previous_utterances$", "")
        return prompt

    def make_con_ul_prompt_cra(self, instance, template):
        if not isinstance(instance["context_str"], str):
            if instance["spkr"] == "A":
                prompt = self.get_con_ul_prompt_cra("Alice: " + instance["utt"] + "\n", None, template)
            elif instance["spkr"] == "B":
                prompt = self.get_con_ul_prompt_cra("Bob: " + instance["utt"] + "\n", None, template)
        else:
            if instance["spkr"] == "A":
                prompt = self.get_con_ul_prompt_cra("Alice: " + instance["utt"] + "\n", instance["context_str"], template)
            elif instance["spkr"] == "B":
                prompt = self.get_con_ul_prompt_cra("Bob: " + instance["utt"] + "\n", instance["context_str"], template)
        return prompt

    def get_prompt_ji(self, instance, template):
        dialogue = ""
        for comment in instance.comments:
            dialogue += comment.user.context["role"] + ": " + comment.body + "\n"
        return template.replace("$dialogue$", dialogue)

    def get_da_dialogue_with_bids_ji(self, instance):
        comments_dict = {}
        for comment in instance.comments:
            comment_str = comment.user.context["role"] + ": " + comment.body + "\n"
            comments_dict[comment.created_at] = comment_str

        bids_dict = {}
        for bid in instance.bids:
            bid_str = bid.user.context["role"] + ": < propose > " + str(bid.options) + "\n"
            if bid.accepted:
                if bid.user.context["role"] == "worker":
                    bid_response_str = "recruiter: < accept bid >\n"
                elif bid.user.context["role"] == "recruiter":
                    bid_response_str = "worker: < accept bid >\n"
            else:
                if bid.user.context["role"] == "worker":
                    bid_response_str = "recruiter: < reject bid >\n"
                elif bid.user.context["role"] == "recruiter":
                    bid_response_str = "worker: < reject bid >\n"
            bids_dict[bid.created_at] = [bid_str, bid_response_str]

        comments_dict.update(bids_dict)
        dialogue_dict = copy.deepcopy(comments_dict)
        list_of_tuples = sorted(dialogue_dict.items())

        dialogue = ""
        for timestamp, string_or_list in list_of_tuples:
            if isinstance(string_or_list, str):
                dialogue += string_or_list
            elif isinstance(string_or_list, list):
                dialogue += string_or_list[0] + string_or_list[1]
        return dialogue

    def get_dialogue_with_bids_ji(self, instance):
        full_dialogue = self.get_da_dialogue_with_bids_ji(instance)
        dialogue_lines = full_dialogue.split("\n")
        dialogue_lines = [line.strip() for line in dialogue_lines if line.strip() != ""]

        if "< accept bid >" in dialogue_lines[-1] or "< reject bid >" in dialogue_lines[-1]:
            assert "< propose >" in dialogue_lines[-2]
            dialogue_lines_final = dialogue_lines[:-2]
        else:
            dialogue_lines_final = dialogue_lines[:]

        if self.args.num_utts_partial_dial != -1:
            dialogue_lines_final = dialogue_lines_final[:self.args.num_utts_partial_dial]

        dialogue = ""
        for line in dialogue_lines_final:
            dialogue += line.strip() + "\n"
        return dialogue

    def get_prompt_with_bids_ji(self, instance, template):
        dialogue = self.get_dialogue_with_bids_ji(instance)
        temp_filled = template.replace("$dialogue$", dialogue)

        agent = "worker"
        if instance.users[0].context["role"] == agent:
            weights_list = {d["name"]: d["weight"] for d in instance.users[0].context["utilities"]}
        elif instance.users[1].context["role"] == agent:
            weights_list = {d["name"]: d["weight"] for d in instance.users[1].context["utilities"]}

        wdict = copy.deepcopy(weights_list)
        wdict["position"] = wdict.pop("Position")
        wdict["company"] = wdict.pop("Company")
        wdict["salary"] = wdict.pop("Salary")
        wdict["days_off"] = wdict.pop("Weekly holiday")
        wdict["workplace"] = wdict.pop("Workplace")

        temp_filled = temp_filled.replace("$pos_weight$", str(round(wdict["position"], 3)))
        temp_filled = temp_filled.replace("$comp_weight$", str(round(wdict["company"], 3)))
        temp_filled = temp_filled.replace("$salary_weight$", str(round(wdict["salary"], 3)))
        temp_filled = temp_filled.replace("$workplace_weight$", str(round(wdict["workplace"], 3)))
        temp_filled = temp_filled.replace("$days_off_weight$", str(round(wdict["days_off"], 3)))
        return temp_filled

    def get_turns_from_dialogue_ji(self, instance):
        dialogue = self.get_da_dialogue_with_bids_ji(instance)
        dialogue_lines = dialogue.split("\n")
        turns_list = []
        if "recruiter:" in dialogue_lines[0]:
            speaker = "recruiter:"
        elif "worker:" in dialogue_lines[0]:
            speaker = "worker:"

        turn = []
        for index in range(len(dialogue_lines)):
            if "< propose >" in dialogue_lines[index] or "< accept bid >" in dialogue_lines[index] or "< reject bid >" in dialogue_lines[index]:
                if turn:
                    turns_list.append(turn)
                    turn = []
                    turn.append(dialogue_lines[index])
                    turns_list.append(turn)
                else:
                    turn.append(dialogue_lines[index])
                    turns_list.append(turn)
                turn = []
                if index != len(dialogue_lines) - 1:
                    if "recruiter:" in dialogue_lines[index + 1]:
                        speaker = "recruiter:"
                    elif "worker:" in dialogue_lines[index + 1]:
                        speaker = "worker:"
            elif speaker in dialogue_lines[index]:
                turn.append(dialogue_lines[index])
            else:
                if turn:
                    turns_list.append(turn)
                    turn = []
                turn.append(dialogue_lines[index])
                if speaker == "worker:":
                    speaker = "recruiter:"
                elif speaker == "recruiter:":
                    speaker = "worker:"

        return turns_list

    def get_all_turns(self, dataset_handler):
        instances = dataset_handler.get_instances()
        num_dialogue = 0
        organized_turns = []
        for instance in instances:
            turns = self.get_turns_from_dialogue_ji(instance)
            for turn in turns:
                organized_turns.append({"num_dialogue": num_dialogue, "turn_comments": turn, "orig_instance": copy.deepcopy(instance)})
            num_dialogue += 1
        return organized_turns

    def get_turns(self, dataset_handler):
        organized_turns = self.get_all_turns(dataset_handler)
        return organized_turns[:self.args.num_instances]

    def get_reg_da_prompt_ji(self, turn, template):
        turn_string = ""
        list_of_phrases = turn["turn_comments"]
        if "worker: " in list_of_phrases[0]:
            speaker = "worker: "
        elif "recruiter: " in turn["turn_comments"][0]:
            speaker = "recruiter: "

        for phrase in list_of_phrases:
            turn_string += phrase.replace("worker: ", "").replace("recruiter: ", "") + "\n"
        turn_string = speaker + turn_string
        temp_filled = template.replace("$utterance$", turn_string)

        agent = "worker"
        instance = turn["orig_instance"]
        if instance.users[0].context["role"] == agent:
            weights_list = {d["name"]: d["weight"] for d in instance.users[0].context["utilities"]}
        elif instance.users[1].context["role"] == agent:
            weights_list = {d["name"]: d["weight"] for d in instance.users[1].context["utilities"]}

        wdict = copy.deepcopy(weights_list)
        wdict["position"] = wdict.pop("Position")
        wdict["company"] = wdict.pop("Company")
        wdict["salary"] = wdict.pop("Salary")
        wdict["days_off"] = wdict.pop("Weekly holiday")
        wdict["workplace"] = wdict.pop("Workplace")

        temp_filled = temp_filled.replace("$pos_weight$", str(round(wdict["position"], 3)))
        temp_filled = temp_filled.replace("$comp_weight$", str(round(wdict["company"], 3)))
        temp_filled = temp_filled.replace("$salary_weight$", str(round(wdict["salary"], 3)))
        temp_filled = temp_filled.replace("$workplace_weight$", str(round(wdict["workplace"], 3)))
        temp_filled = temp_filled.replace("$days_off_weight$", str(round(wdict["days_off"], 3)))
        return temp_filled

    def get_con_da_prompt_ji(self, turn, prev_turn, template):
        base_prompt = self.get_reg_da_prompt_ji(turn, template)
        if prev_turn:
            prev_turn_string = ""
            prev_list_of_phrases = prev_turn["turn_comments"]
            if "worker: " in prev_list_of_phrases[0]:
                prev_speaker = "worker: "
            elif "recruiter: " in prev_list_of_phrases[0]:
                prev_speaker = "recruiter: "
            for phrase in prev_list_of_phrases:
                prev_turn_string += phrase.replace("worker: ", "").replace("recruiter: ", "") + "\n"
            prev_turn_string = prev_speaker + prev_turn_string
            prompt = base_prompt.replace("$previous_utterance$", prev_turn_string)
        else:
            prompt = base_prompt.replace("$previous_utterance$", "None")
        return prompt

    def get_turn_dialogues(self, dataset_handler):
        organized_turns = self.get_all_turns(dataset_handler)
        dialogue_counter = 0
        dialogues = []
        dialogue = ""

        for turn in organized_turns:
            if turn["num_dialogue"] == dialogue_counter:
                turn_comments = turn["turn_comments"]
                for index in range(len(turn_comments)):
                    if index == 0:
                        dialogue += turn_comments[index] + "  "
                    else:
                        dialogue += turn_comments[index].replace("worker: ", "").replace("recruiter: ", "") + "  "
                dialogue += "\n"
            else:
                dialogues.append(dialogue)
                dialogue_counter += 1
                dialogue = ""
                turn_comments = turn["turn_comments"]
                for index in range(len(turn_comments)):
                    if index == 0:
                        dialogue += turn_comments[index] + "  "
                    else:
                        dialogue += turn_comments[index].replace("worker: ", "").replace("recruiter: ", "") + "  "
                dialogue += "\n"
        dialogues.append(dialogue)
        return dialogues

    def get_da_dialogue_prompt_ji(self, dialogue, template):
        dialogue = dialogue.replace("'", "")
        return template.replace("$dialogue$", dialogue)

    def get_final_outputs(self, outputs_dict, possible_outputs, prompts, ground_truth, cot_bool=False):
        final_prompts, final_predictions, final_ground_truth = [], [], []

        for prompt, gt in zip(prompts, ground_truth):
            if prompt not in outputs_dict:
                continue
            final_prompts.append(prompt)
            final_ground_truth.append(gt)
            raw = outputs_dict[prompt]

            inner = _extract_last_answer(raw)
            if inner is not None:
                answer = inner
            elif cot_bool:
                final_predictions.append("Manual prediction extraction required.")
                continue
            else:
                answer = raw

            if answer.strip() in possible_outputs:
                final_predictions.append(answer.strip())
            elif any([po in answer.replace("YOU", "a").replace("THEM", "a") for po in possible_outputs]):
                list_of_words = []
                for word in answer.replace("YOU", "a").replace("THEM", "a").split(" "):
                    flags_for_word = [po for po in possible_outputs if po in word]
                    if flags_for_word:
                        list_of_words.append(flags_for_word[-1])
                final_predictions.append(list_of_words[-1])
            else:
                final_predictions.append("Manual prediction extraction required.")

        return final_prompts, final_predictions, final_ground_truth

    def get_final_outputs_ann(self, outputs_dict, labels, prompts, ground_truth):
        final_prompts, final_predictions, final_ground_truth = [], [], []

        for prompt, gt in zip(prompts, ground_truth):
            if prompt not in outputs_dict:
                continue
            final_prompts.append(prompt)
            final_ground_truth.append(gt)
            answer = outputs_dict[prompt]

            this_utterance_pred = []
            for label in labels:
                if label in answer:
                    this_utterance_pred.append(label)

            if this_utterance_pred:
                final_predictions.append(this_utterance_pred)
            else:
                final_predictions.append('Manual prediction extraction required. If no labels are found, please replace this with the Python list ["none"]. Ensure the none is in double quotes.')

        return final_prompts, final_predictions, final_ground_truth

    def get_final_outputs_dict(self, outputs_dict, possible_keys, possible_outputs, prompts, ground_truth):
        final_prompts, final_predictions, final_ground_truth = [], [], []

        for prompt, gt in zip(prompts, ground_truth):
            if prompt not in outputs_dict:
                continue
            final_prompts.append(prompt)
            final_ground_truth.append(gt)

            pred = {}
            raw = outputs_dict[prompt]
            inner = _extract_last_json_answer(raw)
            if inner is None:
                inner = _extract_last_answer(raw) or ""

            try:
                answer = inner.replace("\n", "").replace(" ", "")
                annotations = json.loads(answer)
                assert len(annotations) == len(possible_keys)
                for k, v in annotations.items():
                    assert k in possible_keys
                    v = str(v)
                    if v in possible_outputs:
                        pred[k] = v
                    elif any([po in v.replace("YOU", "a").replace("THEM", "a") for po in possible_outputs]):
                        list_of_words = []
                        for word in v.replace("YOU", "a").replace("THEM", "a").split(" "):
                            flags_for_word = [po for po in possible_outputs if po in word]
                            if flags_for_word:
                                list_of_words.append(flags_for_word[-1])
                        pred[k] = list_of_words[-1]
                    else:
                        raise ValueError
                final_predictions.append(pred)
            except Exception:
                final_predictions.append("Manual prediction extraction required.")

        return final_prompts, final_predictions, final_ground_truth

    def log_everything(self, stats, prompts, predictions, ground_truth, outputs_dict, dataset_handler, model_handler):
        if model_handler.multishot:
            storage = {
                "ground truth": ground_truth[2:],
                "predictions": predictions,
                "prompts": prompts,
                "outputs_dict": outputs_dict,
            }
        else:
            storage = {
                "stats": stats,
                "ground truth": ground_truth,
                "predictions": predictions,
                "prompts": prompts,
                "outputs_dict": outputs_dict,
            }

        mname = model_handler.name
        if model_handler.name == "hf_model":
            mname = model_handler.args.hf_model_str.replace("/", "_")
        elif model_handler.name == "open_ai":
            mname = model_handler.args.openai_model_str
        elif model_handler.name == "local_model":
            mname = model_handler.display_name

        out_path = utils.get_output_path(
            self.args.storage_dir, dataset_handler.name, mname,
            self.name, self.args.num_instances, args=self.args
        )
        utils.write_json(storage, out_path)

    def remove_duplicates(self, prompts, ground_truth):
        assert len(prompts) == len(ground_truth)
        list_of_tuples = zip(prompts, ground_truth)
        unique_tuples = []
        seen_first_elements = []
        for t in list_of_tuples:
            if t[0] not in seen_first_elements:
                unique_tuples.append(t)
                seen_first_elements.append(t[0])
        new_prompts = [t[0] for t in unique_tuples]
        new_ground_truth = [t[1] for t in unique_tuples]
        return new_prompts, new_ground_truth


class WBaseTaskHandler:
    """Base handler for every task (W-variant)."""

    def __init__(self, name, args):
        self.name = name
        self.args = args

    def evaluate(self, dataset_handler, model_handler, return_prompt_gt=False):
        raise NotImplementedError

    def get_prompt_ca(self, instance, template, agent):
        dialogue = ""
        logs = instance['chat_logs']
        participant_info = instance['participant_info']

        for log in logs:
            if log['text'] in ["Submit-Deal", "Accept-Deal", "Walk-Away", "Reject-Deal"]:
                continue
            round_str = log['id'] + ": " + log['text'] + "\n"
            dialogue += round_str

        dialogue = dialogue.replace("mturk_agent_1:", "YOU:")
        dialogue = dialogue.replace("mturk_agent_2:", "THEM:")

        agent1_dict = participant_info["mturk_agent_1"]["value2issue"]
        agent2_dict = participant_info["mturk_agent_2"]["value2issue"]
        agent1_switched = {item: level for level, item in agent1_dict.items()}
        agent2_switched = {item: level for level, item in agent2_dict.items()}

        def priority2points(d):
            for k, v in d.items():
                if v == 'Low':
                    d[k] = 3
                elif v == 'Medium':
                    d[k] = 4
                else:
                    d[k] = 5
            return d

        agent1_points = priority2points(agent1_switched)
        agent2_points = priority2points(agent2_switched)

        prompt = template.replace("$dialogue$", dialogue)
        if agent == "mturk_agent_1":
            prompt = prompt.replace("$food_points$", str(agent1_points['Food']))
            prompt = prompt.replace("$water_points$", str(agent1_points['Water']))
            prompt = prompt.replace("$fire_points$", str(agent1_points['Firewood']))
        elif agent == "mturk_agent_2":
            prompt = prompt.replace("$food_points$", str(agent2_points['Food']))
            prompt = prompt.replace("$water_points$", str(agent2_points['Water']))
            prompt = prompt.replace("$fire_points$", str(agent2_points['Firewood']))

        return prompt

    def get_prompt_dnd(self, instance, template, agent):
        dialogue = ""
        dialogue_list = str(instance['dialogue']).split(" <eos> ")
        for turn in dialogue_list[:-1]:
            dialogue += turn + "\n"

        you_value = instance['input']['value']
        them_value = instance['partner_input']['value']
        counts = instance['input']['count']

        prompt = template.replace("$dialogue$", dialogue)
        prompt = prompt.replace("$num_books$", str(counts[0]))
        prompt = prompt.replace("$num_hats$", str(counts[1]))
        prompt = prompt.replace("$num_balls$", str(counts[2]))

        if agent == "YOU":
            prompt = prompt.replace("$book_points$", str(you_value[0]))
            prompt = prompt.replace("$hat_points$", str(you_value[1]))
            prompt = prompt.replace("$ball_points$", str(you_value[2]))
        elif agent == "THEM":
            prompt = prompt.replace("$book_points$", str(them_value[0]))
            prompt = prompt.replace("$hat_points$", str(them_value[1]))
            prompt = prompt.replace("$ball_points$", str(them_value[2]))

        return prompt

    def get_prompt_da_dnd(self, instance, template):
        dialogue = ""
        for i in range(len(instance['events'])):
            if not isinstance(instance['events'][i]['data'], dict):
                if instance['events'][i]['agent'] == 0:
                    dialogue += "Agent 1: " + instance['events'][i]['data'] + "\n"
                else:
                    dialogue += "Agent 2: " + instance['events'][i]['data'] + "\n"
        return template.replace("$dialogue$", dialogue)

    def get_prompt_with_bids_ji(self, instance, template):
        comments_dict = {}
        for comment in instance.comments:
            comment_str = comment.user.context["role"] + ": " + comment.body + "\n"
            comments_dict[comment.created_at] = comment_str
        bids_dict = {}
        for bid in instance.bids:
            bid_str = bid.user.context["role"] + ": < propose > " + str(bid.options) + "\n"
            if bid.accepted:
                if bid.user.context["role"] == "worker":
                    bid_response_str = "recruiter: < accept bid >\n"
                elif bid.user.context["role"] == "recruiter":
                    bid_response_str = "worker: < accept bid >\n"
            else:
                if bid.user.context["role"] == "worker":
                    bid_response_str = "recruiter: < reject bid >\n"
                elif bid.user.context["role"] == "recruiter":
                    bid_response_str = "worker: < reject bid >\n"
            bids_dict[bid.created_at] = [bid_str, bid_response_str]

        comments_dict.update(bids_dict)
        dialogue_dict = copy.deepcopy(comments_dict)
        list_of_tuples = sorted(dialogue_dict.items())

        full_dialogue = ""
        for timestamp, string_or_list in list_of_tuples:
            if isinstance(string_or_list, str):
                full_dialogue += string_or_list
            elif isinstance(string_or_list, list):
                full_dialogue += string_or_list[0] + string_or_list[1]

        dialogue_lines = full_dialogue.split("\n")
        dialogue_lines = [line.strip() for line in dialogue_lines if line.strip() != ""]

        if "< accept bid >" in dialogue_lines[-1] or "< reject bid >" in dialogue_lines[-1]:
            assert "< propose >" in dialogue_lines[-2]
            dialogue_lines_final = dialogue_lines[:-2]
        else:
            dialogue_lines_final = dialogue_lines[:]

        dialogue = ""
        for line in dialogue_lines_final:
            dialogue += line + "\n"

        temp_filled = template.replace("$dialogue$", dialogue)

        agent = "worker"
        if instance.users[0].context["role"] == agent:
            weights_list = {d["name"]: d["weight"] for d in instance.users[0].context["utilities"]}
        elif instance.users[1].context["role"] == agent:
            weights_list = {d["name"]: d["weight"] for d in instance.users[1].context["utilities"]}

        wdict = copy.deepcopy(weights_list)
        wdict["position"] = wdict.pop("Position")
        wdict["company"] = wdict.pop("Company")
        wdict["salary"] = wdict.pop("Salary")
        wdict["days_off"] = wdict.pop("Weekly holiday")
        wdict["workplace"] = wdict.pop("Workplace")

        temp_filled = temp_filled.replace("$pos_weight$", str(round(wdict["position"], 3)))
        temp_filled = temp_filled.replace("$comp_weight$", str(round(wdict["company"], 3)))
        temp_filled = temp_filled.replace("$salary_weight$", str(round(wdict["salary"], 3)))
        temp_filled = temp_filled.replace("$workplace_weight$", str(round(wdict["workplace"], 3)))
        temp_filled = temp_filled.replace("$days_off_weight$", str(round(wdict["days_off"], 3)))
        return temp_filled

    def log_everything(self, stats, prompts, predictions, ground_truth, outputs_dict, dataset_handler, model_handler):
        if model_handler.multishot:
            storage = {
                "ground truth": ground_truth[2:],
                "predictions": predictions,
                "prompts": prompts,
                "outputs_dict": outputs_dict,
            }
        else:
            storage = {
                "stats": stats,
                "ground truth": ground_truth,
                "predictions": predictions,
                "prompts": prompts,
                "outputs_dict": outputs_dict,
            }

        mname = model_handler.name
        if model_handler.name == "hf_model":
            mname = model_handler.args.hf_model_str.replace("/", "_")
        elif model_handler.name == "open_ai":
            mname = model_handler.args.openai_model_str
        elif model_handler.name == "local_model":
            mname = model_handler.display_name

        out_path = utils.get_output_path(
            self.args.storage_dir, dataset_handler.name, mname,
            self.name, self.args.num_instances, args=self.args
        )
        utils.write_json(storage, out_path)

    def get_final_outputs(self, outputs_dict, possible_outputs, prompts, ground_truth, cot_bool=False):
        final_prompts, final_predictions, final_ground_truth = [], [], []

        for prompt, gt in zip(prompts, ground_truth):
            if prompt not in outputs_dict:
                continue
            final_prompts.append(prompt)
            final_ground_truth.append(gt)
            raw = outputs_dict[prompt]

            inner = _extract_last_answer(raw)
            if inner is not None:
                answer = inner
            elif cot_bool:
                final_predictions.append("Manual prediction extraction required.")
                continue
            else:
                answer = raw

            if answer.strip() in possible_outputs:
                final_predictions.append(answer.strip())
            elif any([po in answer.replace("YOU", "a").replace("THEM", "a") for po in possible_outputs]):
                list_of_words = []
                for word in answer.replace("YOU", "a").replace("THEM", "a").split(" "):
                    flags_for_word = [po for po in possible_outputs if po in word]
                    if flags_for_word:
                        list_of_words.append(flags_for_word[-1])
                final_predictions.append(list_of_words[-1])
            else:
                final_predictions.append("Manual prediction extraction required.")

        return final_prompts, final_predictions, final_ground_truth

    def get_final_outputs_dict(self, outputs_dict, possible_keys, possible_outputs, prompts, ground_truth):
        final_prompts, final_predictions, final_ground_truth = [], [], []

        for prompt, gt in zip(prompts, ground_truth):
            if prompt not in outputs_dict:
                continue
            final_prompts.append(prompt)
            final_ground_truth.append(gt)

            pred = {}
            raw = outputs_dict[prompt]
            inner = _extract_last_json_answer(raw)
            if inner is None:
                inner = _extract_last_answer(raw) or ""

            try:
                answer = inner.replace("\n", "").replace(" ", "")
                annotations = json.loads(answer)
                assert len(annotations) == len(possible_keys)
                for k, v in annotations.items():
                    assert k in possible_keys
                    v = str(v)
                    if v in possible_outputs:
                        pred[k] = v
                    elif any([po in v.replace("YOU", "a").replace("THEM", "a") for po in possible_outputs]):
                        list_of_words = []
                        for word in v.replace("YOU", "a").replace("THEM", "a").split(" "):
                            flags_for_word = [po for po in possible_outputs if po in word]
                            if flags_for_word:
                                list_of_words.append(flags_for_word[-1])
                        pred[k] = list_of_words[-1]
                    else:
                        raise ValueError
                final_predictions.append(pred)
            except Exception:
                final_predictions.append("Manual prediction extraction required.")

        return final_prompts, final_predictions, final_ground_truth

    def get_partial_dial_ca(self, num_utt, instance, prompt_template):
        dialogue = ""
        logs = instance['chat_logs']
        participant_info = instance['participant_info']
        history = logs[:num_utt]

        history2 = []
        for utt in history:
            if utt['text'] not in ["Submit-Deal", "Accept-Deal", "Walk-Away", "Reject-Deal"]:
                history2.append(utt)
        if len(history2) > 5:
            history2 = history2[-5:]

        for utt in history2:
            dialogue += utt['id'] + ": " + utt['text'] + "\n"
        dialogue = dialogue.replace("mturk_agent_1:", "YOU:")
        dialogue = dialogue.replace("mturk_agent_2:", "THEM:")

        agent1_dict = participant_info["mturk_agent_1"]["value2issue"]
        agent1_switched = {item: level for level, item in agent1_dict.items()}

        def priority2points(d):
            for k, v in d.items():
                if v == 'Low':
                    d[k] = 3
                elif v == 'Medium':
                    d[k] = 4
                else:
                    d[k] = 5
            return d

        agent1_points = priority2points(agent1_switched)
        prompt = prompt_template.replace("$dialogue$", dialogue)
        prompt = prompt.replace("$food_points$", str(agent1_points['Food']))
        prompt = prompt.replace("$water_points$", str(agent1_points['Water']))
        prompt = prompt.replace("$fire_points$", str(agent1_points['Firewood']))
        return prompt

    def get_partial_dial_dnd(self, num_utt, instance, dialogue_list, prompt_template):
        dialogue = ""
        history = dialogue_list[:num_utt]
        if len(history) > 5:
            history = history[-5:]
        for utt in history:
            dialogue += utt.strip() + "\n"

        value = instance['input']['value']
        counts = instance['input']['count']

        prompt = prompt_template.replace("$dialogue$", dialogue)
        prompt = prompt.replace("$num_books$", str(counts[0]))
        prompt = prompt.replace("$num_hats$", str(counts[1]))
        prompt = prompt.replace("$num_balls$", str(counts[2]))
        prompt = prompt.replace("$book_points$", str(value[0]))
        prompt = prompt.replace("$hat_points$", str(value[1]))
        prompt = prompt.replace("$ball_points$", str(value[2]))
        return prompt

    def remove_duplicates(self, prompts, ground_truth):
        assert len(prompts) == len(ground_truth)
        list_of_tuples = zip(prompts, ground_truth)
        unique_tuples = []
        seen_first_elements = []
        for t in list_of_tuples:
            if t[0] not in seen_first_elements:
                unique_tuples.append(t)
                seen_first_elements.append(t[0])
        new_prompts = [t[0] for t in unique_tuples]
        new_ground_truth = [t[1] for t in unique_tuples]
        return new_prompts, new_ground_truth
