"""Strategy quality reward using a fine-tuned Flan-T5-small classifier."""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rl.handlers.casino.dataset import STRATEGY_LABELS

LABEL_LIST = sorted(STRATEGY_LABELS)
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


class StrategyClassifier:
    """Wraps a fine-tuned Flan-T5-small multi-label classifier."""

    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> set[str]:
        """Return the set of predicted strategy labels for an utterance."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        ).to(self.device)
        logits = self.model(**inputs).logits
        probs = torch.sigmoid(logits[0])
        return {ID2LABEL[i] for i, p in enumerate(probs) if p > 0.5}


def strategy_reward(
    talk: str,
    turn_index: int,
    classifier: StrategyClassifier,
) -> float:
    """Score the agent's talk based on strategy appropriateness.

    Early turns (0-3) should use information-gathering strategies
    (elicit-pref). Mid turns should use self-need strategies.
    """
    predicted = classifier.predict(talk)
    if not predicted:
        return 0.0

    score = 0.0

    if turn_index <= 3:
        if "elicit-pref" in predicted:
            score += 0.5
        if "small-talk" in predicted:
            score += 0.2
    else:
        if "self-need" in predicted:
            score += 0.5
        if "vouch-fair" in predicted:
            score += 0.2

    if "coordination" in predicted:
        score += 0.1

    return min(score, 1.0)
