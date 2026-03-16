"""
CRA dataset handler.

Source: https://www.researchgate.net/publication/295854474
"""

import os
import copy
import pandas as pd
from .base import BaseDatasetHandler


def _get_data_root():
    return os.path.join(os.path.dirname(__file__), "..", "..", "data", "cra")


class CRAHandler(BaseDatasetHandler):
    """Handler for the CRA dataset."""

    def setup_dataset(self):
        data_root = _get_data_root()
        df = pd.read_csv(os.path.join(data_root, "cra.test.csv"))
        list_of_dicts = df.to_dict(orient='records')
        self.dataset = []

        for item in list_of_dicts:
            item2 = copy.deepcopy(item)
            item2["DUD"] = item2["DUD"].replace("\"", '')
            item2["DIV"] = item2["DIV"].replace("\"", '')
            item2["spkr"] = item2["spkr"].replace("\"", '')
            self.dataset.append(item2)

    def get_instances(self):
        return self.dataset

    def get_ground_truth(self):
        instances = self.get_instances()
        ground_truth = []
        for instance in instances:
            instance_da = []
            if instance["make_offer"] == 1:
                instance_da.append("make offer")
            if instance["ask_offer"] == 1:
                instance_da.append("ask offer")
            if instance["accept"] == 1:
                instance_da.append("accept")
            if instance["reject"] == 1:
                instance_da.append("reject")
            if instance["ask_preference"] == 1:
                instance_da.append("ask preference")
            if instance["share_preference"] == 1:
                instance_da.append("share preference")
            if not instance_da:
                instance_da.append("none")
            ground_truth.append(instance_da)
        return ground_truth

    def get_da_instances(self):
        instances = self.get_instances()
        ground_truth = self.get_ground_truth()
        assert len(instances) == len(ground_truth)

        da_instances = []
        for index in range(len(instances)):
            if ground_truth[index] != ["none"]:
                da_instances.append(instances[index])

        return da_instances[:self.args.num_instances]

    def get_da_ground_truth(self):
        ground_truth = self.get_ground_truth()
        da_ground_truth = [g for g in ground_truth if g != ["none"]]
        return da_ground_truth[:self.args.num_instances]

    def get_slot_instances(self):
        instances = self.get_instances()
        slot_instances = []
        for instance in instances:
            if instance["DIV"] != "[]" and ":" not in instance["DIV"] and instance["DUD"] != "[]":
                slot_instances.append(instance)
        return slot_instances[:self.args.num_instances]

    def get_utt_template(self, context_bool, da_bool, cot_bool):
        template = "Task Description: You are negotiating with a partner over 1 painting, 2 lamps, and 3 records to determine who gets which items. Different types of items are worth different amount of points to each one of you. You'll be provided with an utterance from the conversation. Then, you'll answer a question."

        if da_bool:
            template += """\n\nHere is a list of dialogue acts, contained in <da> tags:\n\n<da>\nmake offer: proposing a full or a partial offer\nask offer: asking the partner to make a full or partial offer\naccept: agreeing to a previous offer\nreject: declining a previous offer\nask preference: asking the partner about which items they prefer\nshare preference: sharing which items you prefer\n</da>"""

        if self.args.num_prior_utts > 0:
            template += """\n\nHere is context for the utterance, contained in <context> tags.\n<context>\n$previous_utterances$\n</context>"""

        template += """\n\nHere is an utterance, contained in <utterance> tags.\n<utterance>\n$utterance$\n</utterance>"""

        template += """\n\nQuestion: $question$"""

        if cot_bool:
            template += """\n\nNOTE: Let's think step-by-step! Put your thoughts in <thinking> </thinking> tags, and put your answer in <answer> </answer> tags. $output_specification$"""
        else:
            template += " $output_specification$"

        return template
