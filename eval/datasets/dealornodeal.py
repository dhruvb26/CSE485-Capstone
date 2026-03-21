"""
DealOrNoDeal dataset handler.

Source: https://huggingface.co/datasets/deal_or_no_dialog
"""

import json
import os
import pandas as pd
from .base import BaseDatasetHandler


def _get_data_root():
    return os.path.join(os.path.dirname(__file__), "..", "..", "data", "dnd")


class DNDHandler(BaseDatasetHandler):
    """Handler for the DealOrNoDeal (DND) dataset."""

    def setup_dataset(self):
        data_root = _get_data_root()

        df = pd.read_csv(os.path.join(data_root, "dnd.test.csv"))
        dnd_sample = df.to_dict(orient="records")

        for inst in dnd_sample:
            inst['input'] = json.loads(inst['input'])
            inst['partner_input'] = json.loads(inst['partner_input'])

        self.dataset = dnd_sample

        df = pd.read_csv(os.path.join(data_root, "dnd_ann.test.csv"))
        ann_sample = df.to_dict(orient="records")

        for inst in ann_sample:
            inst['agents'] = json.loads(inst['agents'])
            inst['agents_info'] = json.loads(inst['agents_info'])
            inst['events'] = json.loads(inst['events'])
            inst['outcome'] = json.loads(inst['outcome'])
            inst['scenario'] = json.loads(inst['scenario'])

        self.annotated_dataset = ann_sample

    def get_instances(self):
        return self.dataset[:self.args.num_instances]

    def get_da_instances(self):
        return self.annotated_dataset[:self.args.num_instances]

    def get_annotated_utterances(self):
        organized_utterances = []
        for instance in self.annotated_dataset:
            for utterance_dict in instance["events"]:
                if utterance_dict["action"] == "message":
                    counts = {dict["Name"]: dict["Count"] for dict in instance["scenario"]["kbs"][0]}
                    you_values = {dict["Name"]: dict["Value"] for dict in instance["scenario"]["kbs"][0]}
                    them_values = {dict["Name"]: dict["Value"] for dict in instance["scenario"]["kbs"][1]}
                    utterance_dict["Counts"] = counts
                    utterance_dict["You_values"] = you_values
                    utterance_dict["Them_values"] = them_values
                    organized_utterances.append(utterance_dict)
            organized_utterances.append("< sep >")

        dialogue_num = 0
        with_dialogue_tags = []
        for utt in organized_utterances:
            if utt != "< sep >":
                utt["dialogue_num"] = dialogue_num
                with_dialogue_tags.append(utt)
            else:
                dialogue_num += 1

        ann_utterances = with_dialogue_tags[:self.args.num_instances]
        return with_dialogue_tags, ann_utterances

    def get_propose_and_extra_utterances(self):
        with_dialogue_tags, _ = self.get_annotated_utterances()

        max_prop_utterances = 0
        for item in with_dialogue_tags:
            if item["metadata"]["intent"] == "propose":
                max_prop_utterances += 1

        prop_utterances = []
        num_prop_utterances = 0
        index = 0

        while num_prop_utterances < min(max_prop_utterances, self.args.num_instances):
            utt = with_dialogue_tags[index]
            if utt["metadata"]["intent"] == "propose":
                num_prop_utterances += 1
            prop_utterances.append(utt)
            index += 1

        return prop_utterances

    def get_dial_template(self, counts_bool, values_bool, dialogue_bool, da_bool, cot_bool, full_dialogue_bool=False):
        template = """Task Description: You are negotiating with a partner over some quantity of books, hats, and balls to determine who gets which items. Different types of items are worth different amount of points to each one of you. You'll be provided with information about the negotiation. Then, you'll answer a question."""

        if counts_bool:
            template += """\n\nHere are the number of books, hats, and balls available in the negotiation, contained in <count> tags.\n<count>\nBooks: $num_books$\nHats: $num_hats$\nBalls: $num_balls$\n</count>"""

        if values_bool:
            template += """\n\nHere are the number of points you get for each type of item, contained in <value> tags.\n<value>\nEach Book: $book_points$ points\nEach Hat: $hat_points$ points\nEach Ball: $ball_points$ points\n</value>"""

        if da_bool:
            template += """\n\nHere are a list of dialogue acts, contained in <da> tags: \n\n<da>\ngreet\ninquire\npropose\nagree\ndisagree\ninsist\nunknown\n</da>"""

        if dialogue_bool:
            template += """\n\nHere is the recent dialogue history, contained in <dialogue> tags.\n<dialogue>\n$dialogue$\n</dialogue>"""

        if full_dialogue_bool:
            template += """\n\nHere is the complete dialogue, contained in <dialogue> tags.\n<dialogue>\n$dialogue$\n</dialogue>"""

        template += """\n\nQuestion: $question$"""

        if cot_bool:
            template += """\n\nNOTE: Let's think step-by-step! Put your thoughts in <thinking> </thinking> tags, and put your answer in <answer> </answer> tags. $output_specification$"""
        else:
            template += " $output_specification$"

        return template

    def get_utt_template(self, counts_bool, values_bool, context_bool, da_bool, cot_bool):
        template = """Task Description: You are negotiating with a partner over some quantity of books, hats, and balls to determine who gets which items. Different types of items are worth different amount of points to each one of you. You'll be provided with information about the negotiation. Then, you'll answer a question."""

        if counts_bool:
            template += """\n\nHere are the number of books, hats, and balls available in the negotiation, contained in <count> tags.\n<count>\nBooks: $num_books$\nHats: $num_hats$\nBalls: $num_balls$\n</count>"""

        if values_bool:
            template += """\n\nHere are the number of points you get for each type of item, contained in <value> tags.\n<value>\nEach Book: $book_points$ points\nEach Hat: $hat_points$ points\nEach Ball: $ball_points$ points\n</value>"""

        if da_bool:
            template += """\n\nHere are a list of dialogue acts, contained in <da> tags: \n\n<da>\ngreet\ninquire\npropose\nagree\ndisagree\ninsist\nunknown\n</da>"""

        if self.args.num_prior_utts > 0:
            template += """\n\nHere is context for the utterance, contained in <context> tags.\n<context>\n$previous_utterance$\n</context>"""

        template += """\n\nHere is an utterance from the negotiation, contained in <utterance> tags.\n<utterance>\n$utterance$\n</utterance>"""

        template += "\n\nQuestion: $question$"

        if cot_bool:
            template += """\n\nNOTE: Let's think step-by-step! Put your thoughts in <thinking> </thinking> tags, and put your answer in <answer> </answer> tags. $output_specification$"""
        else:
            template += " $output_specification$"

        return template
