"""Negotiation environment abstraction — dataset-specific logic for self-play."""

from __future__ import annotations

import ast
import csv
import json
import re
from abc import ABC, abstractmethod


class NegotiateEnv(ABC):
    """Interface that dataset-specific negotiation envs must implement."""

    @abstractmethod
    def load_scenarios(self, csv_path: str) -> list[dict]: ...

    @abstractmethod
    def build_system_prompt(self, scenario: dict, agent_id: str) -> str: ...

    @abstractmethod
    def parse_deal(self, action_text: str) -> dict[str, int] | None: ...

    @abstractmethod
    def flip_deal(self, text: str, scenario: dict) -> str: ...

    @abstractmethod
    def compute_points(
        self, deal: dict[str, int], scenario: dict, agent_id: str
    ) -> int: ...

    @abstractmethod
    def validate_deal(self, deal: dict[str, int], scenario: dict) -> bool: ...

    @abstractmethod
    def invert_alloc(self, deal: dict[str, int], scenario: dict) -> dict[str, int]: ...


_CASINO_POINTS = {"High": 5, "Medium": 4, "Low": 3}

_CASINO_SYSTEM_PROMPT = """\
You are negotiating with your campsite neighbor over extra supply of food, water, and firewood for your camping trip.

There are exactly 3 packages of each item (food, water, firewood) to divide between you and your neighbor. Each item allocation in a deal must be between 0 and 3, and the two parties' allocations for each item must sum to 3.

Your items and priorities are:

{items_block}

Your reply must always include all 3 parts in this order:

Thought: your inner strategic thinking of this bargaining session.

Talk: short talk that you are going to say to the neighbor. Speak concisely and cut to the chase. Generate authentic and diverse sentences, avoiding repetition of sentences that have already appeared in the conversation.

Action: one of: [TALK] | [SUBMIT_DEAL] food:F water:W firewood:FW | [ACCEPT_DEAL] | [REJECT_DEAL] | [WALK_AWAY]

Note: When using [SUBMIT_DEAL], specify only YOUR allocation. Your neighbor receives the remainder (since totals must sum to 3 for each item).
When your neighbor proposes a [SUBMIT_DEAL], the values shown represent YOUR allocation — what you would receive.

Here are some examples of completions:

Example 1 — opening with talk:

Thought: They haven't proposed yet. I'll ask what they need before proposing.

Talk: Hi! I'm happy to work something out. What do you need most for your trip?

Action: [TALK]

Example 2 — proposing a deal:

Thought: I want to maximize my top priority. A split of 3 food, 2 water, 1 firewood gives me good points. I'll propose that.

Talk: How about I take 3 food, 2 water, and 1 firewood — you get the rest?

Action: [SUBMIT_DEAL] food:3 water:2 firewood:1

Example 3 — accepting a deal:

Thought: Their offer meets my needs. The split is acceptable.

Talk: That works for me. Let's do it.

Action: [ACCEPT_DEAL]

Example 4 — rejecting and countering:

Thought: Too little of what I need. I'll reject and ask for more.

Talk: I need more than that. Can you give me an extra package?

Action: [REJECT_DEAL]

Example 5 — evaluating a neighbor's offer:

Thought: Their offer gives me food:1 water:0 firewood:2. That's 1x3 + 0x5 + 2x4 = 11 points. I can do better — I'll reject.

Talk: That doesn't work for me. I need more water.

Action: [REJECT_DEAL]"""

_CASINO_DEAL_RE = re.compile(
    r"\[SUBMIT_DEAL\]\s*food:(\d+)\s*water:(\d+)\s*firewood:(\d+)", re.IGNORECASE
)


class CasinoEnv(NegotiateEnv):
    """CaSiNo: food/water/firewood, 3 packages each, High/Medium/Low priorities."""

    def load_scenarios(self, csv_path: str) -> list[dict]:
        scenarios = []
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pi = ast.literal_eval(row["participant_info"])
                scenarios.append({"participant_info": pi, "agent_ids": list(pi.keys())})
        return scenarios

    def build_system_prompt(self, scenario: dict, agent_id: str) -> str:
        pi = scenario["participant_info"]
        v2i = pi[agent_id]["value2issue"]
        v2r = pi[agent_id]["value2reason"]
        items_block = "\n  ".join(
            f"{v2i[p]} ({_CASINO_POINTS[p]} points) - {v2r[p]}" for p in _CASINO_POINTS
        )
        return _CASINO_SYSTEM_PROMPT.format(items_block=items_block)

    def parse_deal(self, action_text: str) -> dict[str, int] | None:
        m = _CASINO_DEAL_RE.search(action_text)
        if m is None:
            return None
        return {
            "food": int(m.group(1)),
            "water": int(m.group(2)),
            "firewood": int(m.group(3)),
        }

    def flip_deal(self, text: str, scenario: dict) -> str:
        def _flip(m: re.Match) -> str:
            f, w, fw = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return f"[SUBMIT_DEAL] food:{3 - f} water:{3 - w} firewood:{3 - fw}"

        return _CASINO_DEAL_RE.sub(_flip, text)

    def compute_points(
        self, deal: dict[str, int], scenario: dict, agent_id: str
    ) -> int:
        pi = scenario["participant_info"]
        point_map = {
            pi[agent_id]["value2issue"][lv].lower(): pts
            for lv, pts in _CASINO_POINTS.items()
        }
        return sum(qty * point_map.get(item.lower(), 0) for item, qty in deal.items())

    def validate_deal(self, deal: dict[str, int], scenario: dict) -> bool:
        return deal is not None and all(0 <= v <= 3 for v in deal.values())

    def invert_alloc(self, deal: dict[str, int], scenario: dict) -> dict[str, int]:
        return {item: 3 - qty for item, qty in deal.items()}


_DND_ITEMS = ("book", "hat", "ball")

_DND_SYSTEM_PROMPT = """\
You are negotiating with your partner over a collection of items.

There are {counts_desc} to divide between you and your partner. Each item allocation in a deal must be between 0 and the total count for that item, and the two parties' allocations for each item must sum to the total.

Your item values:

{items_block}

Your reply must always include all 3 parts in this order:

Thought: your inner strategic thinking of this negotiation session.

Talk: short talk that you are going to say to your partner. Speak concisely and cut to the chase. Generate authentic and diverse sentences, avoiding repetition of sentences that have already appeared in the conversation.

Action: one of: [TALK] | [SUBMIT_DEAL] book:B hat:H ball:BA | [ACCEPT_DEAL] | [REJECT_DEAL] | [WALK_AWAY]

Note: When using [SUBMIT_DEAL], specify only YOUR allocation. Your partner receives the remainder.
When your partner proposes a [SUBMIT_DEAL], the values shown represent YOUR allocation — what you would receive.

Here are some examples of completions:

Example 1 — opening with talk:

Thought: They haven't proposed yet. I'll ask what they value before proposing.

Talk: Hi! Let's figure out a good split. What items are most important to you?

Action: [TALK]

Example 2 — proposing a deal:

Thought: I value books the most at {example_val} pts each. I'll try to claim all of them.

Talk: How about I take all the books, and you can have the hats and balls?

Action: [SUBMIT_DEAL] book:{example_book_count} hat:0 ball:0

Example 3 — accepting a deal:

Thought: Their offer gives me good value. I'll accept.

Talk: That works for me. Deal!

Action: [ACCEPT_DEAL]

Example 4 — rejecting and countering:

Thought: Their offer gives me too few points. I'll reject and ask for more.

Talk: I need a better deal. Can I get at least one more hat?

Action: [REJECT_DEAL]"""

_DND_DEAL_RE = re.compile(
    r"\[SUBMIT_DEAL\]\s*book:(\d+)\s*hat:(\d+)\s*ball:(\d+)", re.IGNORECASE
)


class DNDEnv(NegotiateEnv):
    """Deal or No Deal: book/hat/ball, variable counts, raw point values."""

    def load_scenarios(self, csv_path: str) -> list[dict]:
        scenarios = []
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        # Consecutive row pairs are the same scenario from each perspective.
        for i in range(0, len(rows), 2):
            row = rows[i]
            my = json.loads(row["input"])
            partner = json.loads(row["partner_input"])
            counts = {
                "book": my["count"][0],
                "hat": my["count"][1],
                "ball": my["count"][2],
            }
            scenarios.append(
                {
                    "counts": counts,
                    "agents": {
                        "agent_0": {
                            "value": {
                                "book": my["value"][0],
                                "hat": my["value"][1],
                                "ball": my["value"][2],
                            }
                        },
                        "agent_1": {
                            "value": {
                                "book": partner["value"][0],
                                "hat": partner["value"][1],
                                "ball": partner["value"][2],
                            }
                        },
                    },
                    "agent_ids": ["agent_0", "agent_1"],
                }
            )
        return scenarios

    def build_system_prompt(self, scenario: dict, agent_id: str) -> str:
        counts = scenario["counts"]
        values = scenario["agents"][agent_id]["value"]

        counts_desc = ", ".join(f"{counts[item]} {item}(s)" for item in _DND_ITEMS)
        items_block = "\n  ".join(
            f"{item}: {values[item]} points each (x{counts[item]} available)"
            for item in _DND_ITEMS
        )

        best_item = max(_DND_ITEMS, key=lambda it: values[it])
        return _DND_SYSTEM_PROMPT.format(
            counts_desc=counts_desc,
            items_block=items_block,
            example_val=values[best_item],
            example_book_count=counts["book"],
        )

    def parse_deal(self, action_text: str) -> dict[str, int] | None:
        m = _DND_DEAL_RE.search(action_text)
        if m is None:
            return None
        return {
            "book": int(m.group(1)),
            "hat": int(m.group(2)),
            "ball": int(m.group(3)),
        }

    def flip_deal(self, text: str, scenario: dict) -> str:
        counts = scenario["counts"]

        def _flip(m: re.Match) -> str:
            b, h, ba = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return (
                f"[SUBMIT_DEAL] book:{counts['book'] - b} "
                f"hat:{counts['hat'] - h} ball:{counts['ball'] - ba}"
            )

        return _DND_DEAL_RE.sub(_flip, text)

    def compute_points(
        self, deal: dict[str, int], scenario: dict, agent_id: str
    ) -> int:
        values = scenario["agents"][agent_id]["value"]
        return sum(deal.get(item, 0) * values.get(item, 0) for item in _DND_ITEMS)

    def validate_deal(self, deal: dict[str, int], scenario: dict) -> bool:
        if deal is None:
            return False
        counts = scenario["counts"]
        return all(0 <= deal.get(item, -1) <= counts[item] for item in _DND_ITEMS)

    def invert_alloc(self, deal: dict[str, int], scenario: dict) -> dict[str, int]:
        counts = scenario["counts"]
        return {item: counts[item] - deal.get(item, 0) for item in _DND_ITEMS}


ENVS: dict[str, type[NegotiateEnv]] = {
    "casino": CasinoEnv,
    "dnd": DNDEnv,
}


def get_env(dataset: str) -> NegotiateEnv:
    cls = ENVS.get(dataset)
    if cls is None:
        raise ValueError(f"Unknown dataset {dataset!r}, expected one of {list(ENVS)}")
    return cls()
