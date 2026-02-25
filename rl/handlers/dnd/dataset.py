"""
DealOrNoDeal dataset handler and shared prompt-building utilities.

Input / partner_input schema:
    {"count": [books, hats, balls], "value": [book_pts, hat_pts, ball_pts]}

Key property of the DND dataset: value arrays always satisfy
    sum(count[i] * value[i]) == 10  (the max achievable points is always 10).
"""

import json

import pandas as pd

from rl.handlers.base import BaseDatasetHandler

_PROMPT_HEADER = (
    "Task Description: You are negotiating with a partner over some quantity of books, "
    "hats, and balls to determine who gets which items. Different types of items are "
    "worth different amount of points to each one of you. You'll be provided with "
    "information about the negotiation. Then, you'll answer a question."
    "\n\nHere are the number of books, hats, and balls available in the negotiation, "
    "contained in <count> tags.\n<count>\nBooks: {num_books}\nHats: {num_hats}\n"
    "Balls: {num_balls}\n</count>"
    "\n\nHere are the number of points you get for each type of item, contained in "
    "<value> tags.\n<value>\nEach Book: {book_pts} points\nEach Hat: {hat_pts} points\n"
    "Each Ball: {ball_pts} points\n</value>"
    "\n\nQuestion: {question}"
)


class DNDDatasetHandler(BaseDatasetHandler):
    """Loads dnd.test.csv (or dnd.train.csv) and parses JSON input fields."""

    def load(self):
        df = pd.read_csv(self.data_path)
        self.dataset = []
        for inst in df.to_dict(orient="records"):
            inst["input"] = json.loads(inst["input"])
            inst["partner_input"] = json.loads(inst["partner_input"])
            self.dataset.append(inst)


def agent_input(instance: dict, agent: str) -> dict:
    """Return the input dict for the given agent ("YOU" or "THEM")."""
    return instance["input"] if agent == "YOU" else instance["partner_input"]


def build_prompt(instance: dict, agent: str, question: str, output_spec: str) -> str:
    inp = agent_input(instance, agent)
    counts = inp["count"]   # [books, hats, balls]
    values = inp["value"]   # [book_pts, hat_pts, ball_pts]
    body = _PROMPT_HEADER.format(
        num_books=counts[0], num_hats=counts[1], num_balls=counts[2],
        book_pts=values[0], hat_pts=values[1], ball_pts=values[2],
        question=question,
    )
    return body + " " + output_spec
