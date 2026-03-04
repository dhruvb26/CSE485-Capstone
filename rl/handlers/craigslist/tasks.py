"""
Evaluation tasks for CraigslistBargain.

mid_action_inference_cl  -- predict the action type of the last agent utterance
mid_price_reasoning_cl   -- predict the price offered in the last agent utterance
"""

from rl.handlers.base import BaseTaskHandler
from rl.handlers.craigslist.dataset import (
    build_prompt,
    extract_prices,
    infer_action,
    parse_turns,
)


class ActionInferenceCL(BaseTaskHandler):
    task_id = "mid_action_inference_cl"

    def build_prompt(self, instance: dict, agent: str) -> str:
        all_turns = parse_turns(instance["input"]) + parse_turns(instance["output"])
        dialogue = "\n".join(f"{t['role'].title()}: {t['text']}" for t in all_turns)
        return build_prompt(
            instance,
            "What type of negotiation action does the LAST utterance represent? "
            "Choose one of: propose, counter, accept, reject.",
            'Inside <answer> put a JSON object with one key: "action" whose value is '
            'exactly one of "propose", "counter", "accept", or "reject".',
            dialogue,
        )

    def ground_truth(self, instance: dict, agent: str) -> str:
        output_turns = parse_turns(instance["output"])
        if not output_turns:
            return "propose"
        last = output_turns[-1]
        successful = instance.get("metadata", {}).get("successful", False)
        is_last = True
        return infer_action(last["text"], is_last, successful)

    def parse_output(self, text: str) -> str | None:
        data = self.extract_json(text)
        if data and "action" in data:
            val = str(data["action"]).lower().strip()
            if val in ("propose", "counter", "accept", "reject"):
                return val
        return None

    def score(self, prediction: str, truth: str) -> float:
        return 1.0 if prediction == truth else 0.0


class PriceReasoningCL(BaseTaskHandler):
    task_id = "mid_price_reasoning_cl"

    def build_prompt(self, instance: dict, agent: str) -> str:
        all_turns = parse_turns(instance["input"]) + parse_turns(instance["output"])
        dialogue = "\n".join(f"{t['role'].title()}: {t['text']}" for t in all_turns)
        return build_prompt(
            instance,
            "What is the last price mentioned by the agent (the one making offers) in "
            "the dialogue?",
            'Inside <answer> put a JSON object with one key: "price" whose value is '
            "a number (the dollar amount).",
            dialogue,
        )

    def ground_truth(self, instance: dict, agent: str) -> str:
        output_turns = parse_turns(instance["output"])
        all_text = " ".join(t["text"] for t in output_turns)
        prices = extract_prices(all_text)
        if prices:
            return str(prices[-1])
        return str(instance["_parsed"]["listing_price"])

    def parse_output(self, text: str) -> str | None:
        data = self.extract_json(text)
        if data and "price" in data:
            try:
                return str(float(data["price"]))
            except (ValueError, TypeError):
                pass
        return None

    def score(self, prediction: str, truth: str) -> float:
        try:
            return 1.0 if abs(float(prediction) - float(truth)) < 0.01 else 0.0
        except (ValueError, TypeError):
            return 0.0


CL_TASK_REGISTRY: dict[str, type[BaseTaskHandler]] = {
    "mid_action_inference_cl": ActionInferenceCL,
    "mid_price_reasoning_cl": PriceReasoningCL,
}
