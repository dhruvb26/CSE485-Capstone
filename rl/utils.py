from __future__ import annotations

import re


def extract_tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def strip_thought(text: str) -> str:
    return re.compile(r"<thought>.*?</thought>\s*", re.DOTALL).sub("", text).strip()


def strip_tags(text: str) -> str:
    return (
        re.compile(r"</?(?:thought|talk|action)>", re.IGNORECASE).sub("", text).strip()
    )


def parse_submit_deal(text: str) -> dict[str, int] | None:
    match = re.compile(
        r"\[SUBMIT_DEAL\]\s*food:(\d+)\s*water:(\d+)\s*firewood:(\d+)", re.IGNORECASE
    ).search(text)
    if match is None:
        return None
    return {
        "food": int(match.group(1)),
        "water": int(match.group(2)),
        "firewood": int(match.group(3)),
    }


def prompt_to_text(prompt) -> str:
    """Flatten a prompt (message list or formatted string) into searchable text."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "\n".join(
            msg.get("content", "") for msg in prompt if isinstance(msg, dict)
        )
    return str(prompt)


def extract_discussed_deal(prompt) -> dict[str, int] | None:
    """Return the most recently discussed deal (agent's own allocation).

    Accepts a message list (``list[dict]``) or a flat string.

    Strategy (in priority order):
      1. Last ``<thought>`` that contains ``N item x M pts`` arithmetic for all
         three items.  The *first* mention of each item is taken (thoughts list
         the agent's allocation before the opponent's).  Opponent thoughts are
         already stripped before being added to learner messages, so every
         ``<thought>`` in the prompt belongs to the learner.
      2. Last ``[SUBMIT_DEAL]`` in **assistant-role messages only**, so that
         opponent deals are not mistaken for the learner's intent.

    Thoughts are checked first because they reflect the latest conversational
    state, whereas a ``[SUBMIT_DEAL]`` might have been rejected earlier.
    """
    if isinstance(prompt, list):
        full_text = "\n".join(
            msg.get("content", "") for msg in prompt if isinstance(msg, dict)
        )
        assistant_text = "\n".join(
            msg.get("content", "")
            for msg in prompt
            if isinstance(msg, dict) and msg.get("role") == "assistant"
        )
    else:
        full_text = str(prompt)
        assistant_text = full_text

    thoughts = list(
        re.finditer(r"<thought>(.*?)</thought>", full_text, re.DOTALL)
    )
    for thought_match in reversed(thoughts):
        deal: dict[str, int] = {}
        for m in re.compile(
            r"(\d+)\s+(food|water|firewood)\s*x\s*\d+\s*pts",
            re.IGNORECASE,
        ).finditer(thought_match.group(1)):
            item = m.group(2).lower()
            if item not in deal:
                deal[item] = int(m.group(1))
        if len(deal) == 3:
            return deal

    submit_matches = list(
        re.finditer(
            r"\[SUBMIT_DEAL\]\s*food:(\d+)\s*water:(\d+)\s*firewood:(\d+)",
            assistant_text,
            re.IGNORECASE,
        )
    )
    if submit_matches:
        m = submit_matches[-1]
        return {
            "food": int(m.group(1)),
            "water": int(m.group(2)),
            "firewood": int(m.group(3)),
        }

    return None


def last_opponent_action_is_submit(prompt) -> bool:
    """True if the opponent's (user-role) most recent action was [SUBMIT_DEAL]."""
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return bool(
                    re.search(r"\[SUBMIT_DEAL\]", msg.get("content", ""), re.IGNORECASE)
                )
        return False

    text = str(prompt)
    user_blocks = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, re.DOTALL)
    if user_blocks:
        return bool(re.search(r"\[SUBMIT_DEAL\]", user_blocks[-1], re.IGNORECASE))
    return False
