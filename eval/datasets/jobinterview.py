"""
JobInterview dataset handler.

The dataset and managing code is part of gucci-j's GitHub repository:
https://github.com/gucci-j/negotiation-breakdown-detection
"""

import os
import pandas as pd
from .base import BaseDatasetHandler
from .negotiation_ji import read_ji_negotiations


def _get_data_root():
    return os.path.join(os.path.dirname(__file__), "..", "..", "data", "ji")


class JIHandler(BaseDatasetHandler):
    """Handler for the JobInterview (JI) dataset."""

    def setup_dataset(self):
        data_root = _get_data_root()
        self.dataset = read_ji_negotiations(os.path.join(data_root, "ji.test.json"))

        df = pd.read_csv(os.path.join(data_root, "ji_dacts.test.csv"))
        self.da_list = df.to_dict(orient='records')

    def get_instances(self):
        return self.dataset[:self.args.num_instances]

    def get_dial_template(self, counts_bool, cot_bool, values_bool=False, dialogue_bool=False, full_dialogue_bool=False):
        template = "Task Description: You are a worker who is negotiating with a recruiter over the issues surrounding a job offer. There are 5 issues to discuss: position, company, salary, workplace, and weekly days off. You both value these issues differently. You'll be provided with information about the negotiation. Then, you'll answer a question."

        if counts_bool:
            template += " There are 4 options for position, 4 options for company, and 4 options for workplace. Salary ranges from $20 to $50, and the number of possible weekly days off ranges from 2 to 5."

        if values_bool:
            template += """\n\nHere are the weights that represent your preference towards each issue in <value> tags.\n<value>\nposition: $pos_weight$\ncompany: $comp_weight$\nsalary: $salary_weight$\nworkplace: $workplace_weight$\ndays_off: $days_off_weight$</value>"""

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

    def get_utt_template(self, context_bool, full_dial_bool, cot_bool, counts_bool=False, values_bool=False, da_bool=False):
        template = "Task Description: You are a worker who is negotiating with a recruiter over the issues surrounding a job offer. There are 5 issues to discuss: position, company, salary, workplace, and weekly days off. You both value these issues differently. You'll be provided with information about the negotiation. Then, you'll answer a question."

        if counts_bool:
            template += " There are 4 options for position, 4 options for company, and 4 options for workplace. Salary ranges from $20 to $50, and the number of possible weekly days off ranges from 2 to 5."

        if values_bool:
            template += """\n\nHere are the weights that represent your preference towards each issue in <value> tags.\n<value>\nposition: $pos_weight$\ncompany: $comp_weight$\nsalary: $salary_weight$\nworkplace: $workplace_weight$\ndays_off: $days_off_weight$</value>"""

        if da_bool:
            template += """\n\nHere is a list of dialogue acts, contained in <da> tags:\n\n<da>\ngreet: greeting the partner.\ninquire: asking an open-ended question.\npropose: suggesting an offer or aspect of an offer.\nagree: agreeing to a previous offer.\ndisagree: declining a previous offer.\ninform: sharing useful information such as what they like/dislike the most.\nunknown: none of the dialogue acts above apply.\n</da>"""

        if self.args.num_prior_utts > 0:
            template += """\n\nHere is context for the utterance, contained in <context> tags.\n<context>\n$previous_utterance$\n</context>"""

        if full_dial_bool:
            template += """\n\nHere is the agents' dialogue, contained in <dialogue> tags.\n<dialogue>\n$dialogue$\n</dialogue>"""
        else:
            template += """\n\nHere is an utterance, contained in <utterance> tags.\n<utterance>\n$utterance$\n</utterance>"""

        template += """\n\nQuestion: $question$"""

        if cot_bool:
            template += """\n\nNOTE: Let's think step-by-step! Put your thoughts in <thinking> </thinking> tags, and put your answer in <answer> </answer> tags. $output_specification$"""
        else:
            template += " $output_specification$"

        return template
