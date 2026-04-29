from __future__ import annotations

import ast
import csv
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

from .prompts import (
    BUYER_SYSTEM_PROMPT,
    CASINO_SYSTEM_PROMPT,
    DND_SYSTEM_PROMPT,
    JI_SYSTEM_PROMPT,
    SELLER_SYSTEM_PROMPT,
)


class NegotiateEnv(ABC):
    """Interface that dataset-specific negotiation envs must implement."""

    @abstractmethod
    def load_scenarios(self, csv_path: str) -> list[dict]: ...

    @abstractmethod
    def build_system_prompt(self, scenario: dict, agent_id: str) -> str: ...

    @abstractmethod
    def parse_deal(self, action_text: str) -> dict | None: ...

    @abstractmethod
    def flip_deal(self, text: str, scenario: dict) -> str: ...

    @abstractmethod
    def compute_points(self, deal: dict, scenario: dict, agent_id: str) -> int: ...

    @abstractmethod
    def validate_deal(self, deal: dict, scenario: dict) -> bool: ...

    @abstractmethod
    def invert_alloc(self, deal: dict, scenario: dict) -> dict: ...


_CASINO_POINTS = {"High": 5, "Medium": 4, "Low": 3}

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
        return CASINO_SYSTEM_PROMPT.format(items_block=items_block)

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
        return DND_SYSTEM_PROMPT.format(
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


_PRICE_DEAL_RE = re.compile(
    r"\[SUBMIT_DEAL\]\s*price:\s*\$?\s*(\d+(?:\.\d{1,2})?)", re.IGNORECASE
)


def _parse_price(price_str: str) -> int:
    return round(float(price_str.replace("$", "").replace(",", "")))


class _PriceEnv(NegotiateEnv):
    """Base for single-issue price negotiation (Amazon, Craigslist)."""

    def parse_deal(self, action_text: str) -> dict | None:
        m = _PRICE_DEAL_RE.search(action_text)
        if m is None:
            return None
        return {"price": round(float(m.group(1)))}

    def flip_deal(self, text: str, scenario: dict) -> str:
        return text

    def compute_points(self, deal: dict, scenario: dict, agent_id: str) -> int:
        price = deal["price"]
        if agent_id == "buyer":
            return scenario["buyer_budget"] - price
        return price - scenario["seller_cost"]

    def validate_deal(self, deal: dict, scenario: dict) -> bool:
        if deal is None:
            return False
        p = deal.get("price")
        return p is not None and p >= 0

    def invert_alloc(self, deal: dict, scenario: dict) -> dict:
        return deal

    def _build_price_prompt(self, scenario: dict, agent_id: str) -> str:
        surplus = scenario["buyer_budget"] - scenario["seller_cost"]
        if agent_id == "buyer":
            example_low = scenario["seller_cost"] + round(surplus * 0.3)
            return BUYER_SYSTEM_PROMPT.format(
                title=scenario["title"],
                category=scenario["category"],
                listing_price=scenario["listing_price"],
                description=scenario["description"],
                buyer_budget=scenario["buyer_budget"],
                example_low=example_low,
            )
        example_high = scenario["buyer_budget"] - round(surplus * 0.1)
        return SELLER_SYSTEM_PROMPT.format(
            title=scenario["title"],
            category=scenario["category"],
            listing_price=scenario["listing_price"],
            description=scenario["description"],
            seller_cost=scenario["seller_cost"],
            example_high=example_high,
        )


class AmazonPriceEnv(_PriceEnv):
    """AmazonHistoryPrice: buyer/seller negotiate a single product price.

    B = list_price (MSRP), C = lowest_price (historical low).
    """

    def load_scenarios(self, data_path: str) -> list[dict]:
        path = Path(data_path)
        products: list[dict] = []
        if path.is_dir():
            for json_file in sorted(path.glob("*.json")):
                with open(json_file, encoding="utf-8") as f:
                    products.extend(json.load(f))
        else:
            with open(path, encoding="utf-8") as f:
                products = json.load(f)

        scenarios = []
        for p in products:
            try:
                list_price = _parse_price(p["list_price"])
                lowest_price = _parse_price(p["lowest_price"])
            except (ValueError, KeyError):
                continue
            if list_price <= lowest_price:
                continue

            desc_parts = []
            if p.get("description"):
                desc_parts.append(p["description"][:300])
            if p.get("features"):
                desc_parts.append("Features: " + p["features"][:200])
            description = "\n".join(desc_parts) if desc_parts else ""

            scenarios.append(
                {
                    "title": p.get("title", "Unknown Product"),
                    "category": p.get("category", ""),
                    "description": description,
                    "listing_price": list_price,
                    "buyer_budget": list_price,
                    "seller_cost": lowest_price,
                    "agent_ids": ["buyer", "seller"],
                }
            )
        return scenarios

    def build_system_prompt(self, scenario: dict, agent_id: str) -> str:
        return self._build_price_prompt(scenario, agent_id)


class CraigslistEnv(_PriceEnv):
    """Craigslist Bargains: buyer/seller negotiate a Craigslist listing price.

    B = buyer_target (from data), C = round(listing_price * 0.5).
    """

    def load_scenarios(self, data_path: str) -> list[dict]:
        with open(data_path, encoding="utf-8") as f:
            dialogues = json.load(f)

        scenarios = []
        for d in dialogues:
            kbs = d["scenario"]["kbs"]
            buyer_kb = next(
                (kb for kb in kbs if kb["personal"]["Role"] == "buyer"), None
            )
            if buyer_kb is None:
                continue

            listing_price = buyer_kb["item"]["Price"]
            buyer_budget = buyer_kb["personal"]["Target"]
            seller_cost = round(listing_price * 0.5)

            if buyer_budget <= seller_cost:
                continue

            title = buyer_kb["item"].get("Title", "")
            raw_desc = buyer_kb["item"].get("Description", [])
            description = " ".join(raw_desc)[:300] if raw_desc else ""
            category = d["scenario"].get("category", "")

            scenarios.append(
                {
                    "title": title,
                    "category": category,
                    "description": description,
                    "listing_price": listing_price,
                    "buyer_budget": buyer_budget,
                    "seller_cost": seller_cost,
                    "agent_ids": ["buyer", "seller"],
                }
            )
        return scenarios

    def build_system_prompt(self, scenario: dict, agent_id: str) -> str:
        return self._build_price_prompt(scenario, agent_id)

_JI_POSITIONS = ("Engineer", "Manager", "Designer", "Sales")
_JI_COMPANIES = ("Google", "Facebook", "Apple", "Amazon")
_JI_WORKPLACES = ("Tokyo", "Seoul", "Beijing", "Sydney")

_JI_DEAL_RE = re.compile(
    r"\[SUBMIT_DEAL\]\s*salary:\s*\$?(\d+)\s+position:\s*(\w+)"
    r"\s+company:\s*(\w+)\s+workplace:\s*(\w+)\s+holiday:\s*(\d+)",
    re.IGNORECASE,
)


def _ji_preferences_block(utilities: list[dict], role: str) -> str:
    lines = []
    for util in utilities:
        name = util["name"]
        pct = f"{util['weight'] * 100:.0f}%"

        if util["type"] == "INTEGER":
            if name == "Salary":
                direction = "higher" if role == "worker" else "lower"
                lines.append(
                    f"  Salary (importance: {pct}): you prefer {direction} salary"
                )
            else:
                direction = "more" if role == "worker" else "fewer"
                lines.append(
                    f"  Days off (importance: {pct}): you prefer {direction} days off"
                )
        elif "relatedTo" in util:
            related = util["relatedTo"]
            combos = sorted(
                util["options"], key=lambda o: o.get("weight", 0), reverse=True
            )
            top = ", ".join(
                f"{o['names'][name]}@{o['names'][related]}" for o in combos[:6]
            )
            lines.append(
                f"  {name} (importance: {pct}): depends on {related} — "
                f"best combos: {top}"
            )
        else:
            opts = sorted(
                util["options"], key=lambda o: o.get("weight", 0), reverse=True
            )
            ranked = " > ".join(o["name"] for o in opts)
            lines.append(f"  {name} (importance: {pct}): preference {ranked}")

    return "\n".join(lines)


class JobInterviewEnv(NegotiateEnv):
    """Job Interview: multi-attribute hybrid negotiation (eval only).

    5 issues: Salary, Position, Company, Workplace, Weekly holiday.
    Position×Company have interdependent utilities.
    """

    def load_scenarios(self, data_path: str) -> list[dict]:
        from eval.datasets.negotiation_ji import User, read_ji_negotiations

        negotiations = read_ji_negotiations(data_path)
        scenarios = []
        for nego in negotiations:
            users: dict[str, User] = {}
            for user in nego.users:
                users[user.context["role"]] = user
            if "worker" not in users or "recruiter" not in users:
                continue
            scenarios.append(
                {
                    "users": users,
                    "agent_ids": ["worker", "recruiter"],
                }
            )
        return scenarios

    def build_system_prompt(self, scenario: dict, agent_id: str) -> str:
        user = scenario["users"][agent_id]
        role = user.context["role"]
        other_role = "recruiter" if role == "worker" else "applicant"
        role_desc = "job applicant" if role == "worker" else "hiring recruiter"
        preferences = _ji_preferences_block(user.context["utilities"], role)
        return JI_SYSTEM_PROMPT.format(
            role_desc=role_desc,
            other_role=other_role,
            preferences_block=preferences,
        )

    def parse_deal(self, action_text: str) -> dict | None:
        m = _JI_DEAL_RE.search(action_text)
        if m is None:
            return None
        return {
            "Salary": int(m.group(1)),
            "Position": m.group(2).capitalize(),
            "Company": m.group(3).capitalize(),
            "Workplace": m.group(4).capitalize(),
            "Weekly holiday": int(m.group(5)),
        }

    def flip_deal(self, text: str, scenario: dict) -> str:
        return text

    def compute_points(self, deal: dict, scenario: dict, agent_id: str) -> int:
        from eval.datasets.negotiation_ji import Bid

        user = scenario["users"][agent_id]
        bid = Bid(None, None, deal, None, None)
        return round(user.calc_score(bid) * 100)

    def validate_deal(self, deal: dict, scenario: dict) -> bool:
        if deal is None:
            return False
        s = deal.get("Salary")
        h = deal.get("Weekly holiday")
        if s is None or h is None:
            return False
        if not (20 <= s <= 50 and 2 <= h <= 6):
            return False
        if deal.get("Position") not in _JI_POSITIONS:
            return False
        if deal.get("Company") not in _JI_COMPANIES:
            return False
        if deal.get("Workplace") not in _JI_WORKPLACES:
            return False
        return True

    def invert_alloc(self, deal: dict, scenario: dict) -> dict:
        return deal


ENVS: dict[str, type[NegotiateEnv]] = {
    "casino": CasinoEnv,
    "dnd": DNDEnv,
    "amazon": AmazonPriceEnv,
    "craigslist": CraigslistEnv,
    "ji": JobInterviewEnv,
}


def get_env(dataset: str) -> NegotiateEnv:
    cls = ENVS.get(dataset)
    if cls is None:
        raise ValueError(f"Unknown dataset {dataset!r}, expected one of {list(ENVS)}")
    return cls()
