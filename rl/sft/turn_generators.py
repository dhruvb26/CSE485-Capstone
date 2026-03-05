"""Turn-level SFT generators for CaSiNo, DND, and CraigslistBargain.

Each generator walks through real dialogues turn-by-turn and produces
rows whose completion uses XML-delimited thought/talk fields with a JSON
action block::

    <thought>...</thought>
    <talk>...</talk>
    <action>{"type": "...", ...}</action>

The ``thought`` field is synthesised (rule-based arithmetic by default;
GPT-augmented when the caller requests it).  ``talk`` comes directly from
the source data, and ``action`` is parsed/inferred from the utterance text
and dataset annotations.
"""

from __future__ import annotations

import json
import logging
import re

from rl.handlers.casino.dataset import (
    PRIORITY_TO_POINTS,
    STRATEGY_LABEL_MAP,
    STRATEGY_LABELS,
    _META_TURNS,
    sanitize_unicode,
)
from rl.handlers.craigslist.dataset import extract_prices, infer_action

logger = logging.getLogger(__name__)

CA_ITEMS = ["food", "water", "firewood"]
CA_COUNTS: dict[str, int] = {"food": 3, "water": 3, "firewood": 3}

DND_ITEMS = ["book", "hat", "ball"]

_NUM_WORDS: dict[str, int] = {
    "zero": 0, "no": 0, "one": 1, "a": 1, "two": 2,
    "three": 3, "four": 4, "five": 5, "all": -1,
}       

_CA_SYSTEM = (
    "You are negotiating with your campsite neighbor over extra supply of "
    "food, water, and firewood for your camping trip.\n\n"
    "Items available: 3 Food, 3 Water, 3 Firewood\n"
    "Your point values: Food={food}pts, Water={water}pts, Firewood={firewood}pts\n"
    "Max possible points: {max_pts}pts"
)

_DND_SYSTEM = (
    "You are negotiating with a partner over books, hats, and balls.\n\n"
    "Items available: {n_book} Book(s), {n_hat} Hat(s), {n_ball} Ball(s)\n"
    "Your point values: Book={book_pts}pts, Hat={hat_pts}pts, Ball={ball_pts}pts\n"
    "Max possible points: {max_pts}pts"
)

_CL_SYSTEM = (
    "You are the {role} in a price negotiation on an online marketplace.\n\n"
    "Product: {product_name}\n"
    "Listing price: ${listing_price:.2f}\n"
    "Category: {category}"
)

_TURN_INSTRUCTION = (
    "\n\nProduce your next negotiation turn using the following XML format:\n"
    "<thought>your internal reasoning (point arithmetic, partner priority estimate, justification)</thought>\n"
    "<talk>your natural language response to your partner</talk>\n"
    '<action>{"type": "offer"|"counter"|"accept"|"reject", ...item allocations for yourself}</action>'
)

_CL_TURN_INSTRUCTION = (
    "\n\nProduce your next negotiation turn using the following XML format:\n"
    "<thought>your internal reasoning (price analysis and strategy)</thought>\n"
    "<talk>your natural language response to your partner</talk>\n"
    '<action>{"type": "propose"|"counter"|"accept"|"reject", "price": <your proposed price>}</action>'
)

def _parse_quantities(
    text: str, items: list[str], counts: dict[str, int],
) -> dict[str, int] | None:
    """Extract item quantities mentioned in negotiation text."""
    lower = text.lower()
    found: dict[str, int] = {}
    for item in items:
        patterns = [
            rf"(\d+)\s+(?:packages?\s+of\s+)?{item}s?",
            rf"(zero|no|one|a|two|three|four|five|all)\s+(?:of\s+the\s+|packages?\s+of\s+)?{item}s?",
        ]
        for pat in patterns:
            m = re.search(pat, lower)
            if m:
                v = m.group(1)
                if v.isdigit():
                    found[item] = int(v)
                elif v in _NUM_WORDS:
                    n = _NUM_WORDS[v]
                    found[item] = counts.get(item, 3) if n == -1 else n
                break
    for item in list(found):
        found[item] = min(found[item], counts.get(item, 3))
    return found or None


_NEGATORS = {"not", "don't", "can't", "cannot", "n't", "no", "never"}


def _has_negation_before(text: str, match_start: int) -> bool:
    window = text[max(0, match_start - 20):match_start].lower().split()
    return bool(set(window) & _NEGATORS)


def _action_type(text: str, has_prior_offer: bool) -> str:
    lower = text.lower()
    reject_kw = (
        "no deal", "reject", "can't agree", "cannot agree",
        "walk away", "no way", "not going to work",
        "don't agree", "do not agree", "can't accept", "cannot accept",
        "don't accept", "do not accept",
    )
    accept_kw = (
        "deal", "accept", "agree", "sounds good", "works for me",
        "that's fine", "okay deal", "ok deal", "you got it",
        "that works", "i can do that",
    )
    if any(k in lower for k in reject_kw):
        return "reject"
    for kw in accept_kw:
        idx = lower.find(kw)
        if idx != -1 and not _has_negation_before(lower, idx):
            return "accept"
    return "counter" if has_prior_offer else "offer"


def _partner_priority_ca(history: list[dict], partner_id: str) -> str:
    scores: dict[str, float] = {i: 0.0 for i in CA_ITEMS}
    need_words = {"need", "want", "important", "essential", "must", "love", "priority", "really"}
    for t in history:
        if t["id"] != partner_id:
            continue
        lower = t["text"].lower()
        w = 2.0 if any(nw in lower for nw in need_words) else 1.0
        for item in CA_ITEMS:
            if item in lower:
                scores[item] += w
    if max(scores.values()) == 0:
        return "unknown"
    return max(scores, key=scores.get)


def _det_thought_multi(
    values: dict[str, int],
    counts: dict[str, int],
    alloc: dict[str, int],
    items: list[str],
    partner_est: str,
    atype: str,
) -> str:
    max_pts = sum(counts[i] * values[i] for i in items)
    my_pts = sum(alloc.get(i, 0) * values[i] for i in items)
    pct = my_pts / max_pts if max_pts > 0 else 0
    vals_str = ", ".join(f"{i}={values[i]}pts" for i in items)
    alloc_str = ", ".join(f"{alloc.get(i, 0)} {i}" for i in items)
    return (
        f"My values: {vals_str}. Max possible = {max_pts}pts. "
        f"My allocation: {alloc_str} = {my_pts}pts ({pct:.0%} of max). "
        f"Partner priority estimate: {partner_est}. Decision: {atype}."
    )


def _fmt_history_ca(chat_logs: list[dict], agent: str, up_to: int) -> str:
    lines: list[str] = []
    for t in chat_logs[:up_to]:
        if t["text"] in _META_TURNS:
            continue
        who = "You" if t["id"] == agent else "Partner"
        lines.append(f"{who}: {sanitize_unicode(t['text'])}")
    return "\n".join(lines) or "(no dialogue yet)"


def _fmt_history_simple(turns: list[dict], agent_id: str, up_to: int) -> str:
    lines: list[str] = []
    for t in turns[:up_to]:
        who = "You" if t["id"] == agent_id else "Partner"
        lines.append(f"{who}: {t['text']}")
    return "\n".join(lines) or "(no dialogue yet)"


def generate_turns_ca(instances: list[dict]) -> list[dict]:
    rows: list[dict] = []

    for inst in instances:
        chat_logs = inst["chat_logs"]
        annotations = inst["annotations"]
        pinfo = inst["participant_info"]

        for agent in ("mturk_agent_1", "mturk_agent_2"):
            partner = "mturk_agent_2" if agent == "mturk_agent_1" else "mturk_agent_1"
            v2i = pinfo[agent]["value2issue"]
            vals = {item.lower(): PRIORITY_TO_POINTS[lv] for lv, item in v2i.items()}
            max_pts = sum(CA_COUNTS[i] * vals[i] for i in CA_ITEMS)
            system = _CA_SYSTEM.format(
                food=vals["food"], water=vals["water"],
                firewood=vals["firewood"], max_pts=max_pts,
            )

            has_prior = False

            for idx, turn in enumerate(chat_logs):
                # ── Final deal (Submit-Deal) ──
                if turn["text"] == "Submit-Deal" and turn["id"] == agent:
                    youget = turn.get("task_data", {}).get("issue2youget", {})
                    if all(v != "" for v in youget.values()):
                        alloc = {k.lower(): int(v) for k, v in youget.items()}
                        hist = _fmt_history_ca(chat_logs, agent, idx)
                        ppri = _partner_priority_ca(chat_logs[:idx], partner)
                        thought = _det_thought_multi(
                            vals, CA_COUNTS, alloc, CA_ITEMS, ppri, "offer (final deal)",
                        )
                        rows.append({
                            "task": "turn_ca",
                            "prompt": system + f"\n\nDialogue so far:\n{hist}" + _TURN_INSTRUCTION,
                            "talk": "I'd like to submit this deal.",
                            "action": {"type": "offer", **alloc},
                            "det_thought": thought,
                            "strategy_label": None,
                        })
                    continue

                if turn["text"] in _META_TURNS or turn["id"] != agent:
                    continue

                # Skip if no prior non-meta turns (greeting only)
                prior_real = [t for t in chat_logs[:idx] if t["text"] not in _META_TURNS]
                if len(prior_real) < 1:
                    continue

                text = sanitize_unicode(turn["text"])
                parsed = _parse_quantities(text, CA_ITEMS, CA_COUNTS)
                atype = _action_type(text, has_prior)

                # Strategy annotation
                strat: str | None = None
                if idx < len(annotations):
                    ann = annotations[idx]
                    raw_label = ann[1] if len(ann) > 1 else ""
                    if raw_label and "non-strategic" not in raw_label:
                        labels = [
                            STRATEGY_LABEL_MAP.get(s.strip(), s.strip())
                            for s in raw_label.split(",")
                        ]
                        labels = [lb for lb in labels if lb in STRATEGY_LABELS]
                        if labels:
                            strat = ", ".join(labels)

                if parsed is not None:
                    alloc = {i: parsed.get(i, 0) for i in CA_ITEMS}
                    has_prior = True
                elif atype in ("accept", "reject"):
                    alloc = {}
                else:
                    continue

                action = {"type": atype, **alloc} if alloc else {"type": atype}
                hist = _fmt_history_ca(chat_logs, agent, idx)
                ppri = _partner_priority_ca(chat_logs[:idx], partner)

                if alloc:
                    thought = _det_thought_multi(vals, CA_COUNTS, alloc, CA_ITEMS, ppri, atype)
                else:
                    thought = f"Partner priority estimate: {ppri}. Decision: {atype}."

                rows.append({
                    "task": "turn_ca",
                    "prompt": system + f"\n\nDialogue so far:\n{hist}" + _TURN_INSTRUCTION,
                    "talk": text,
                    "action": action,
                    "det_thought": thought,
                    "strategy_label": strat,
                })

    return rows


def _parse_dnd_dialogue(dialogue_str: str) -> list[dict]:
    parts = dialogue_str.split("<eos>")
    turns: list[dict] = []
    for part in parts:
        part = part.strip()
        if not part or "<selection>" in part:
            continue
        m = re.match(r"(YOU|THEM):\s*(.*)", part, re.DOTALL)
        if m:
            turns.append({"id": m.group(1), "text": m.group(2).strip()})
    return turns


def _parse_dnd_output(output_str: str) -> tuple[dict[str, int], dict[str, int]]:
    """Parse 'item0=X item1=Y ...' → (you_alloc, them_alloc)."""
    vals: list[int] = []
    for tok in output_str.strip().split():
        if "=" in tok:
            vals.append(int(tok.split("=")[1]))
    if len(vals) < 6:
        logger.warning("Malformed DND output (expected 6 values, got %d): %s", len(vals), output_str[:100])
    you = {"book": vals[0], "hat": vals[1], "ball": vals[2]} if len(vals) >= 3 else {}
    them = {"book": vals[3], "hat": vals[4], "ball": vals[5]} if len(vals) >= 6 else {}
    return you, them


def generate_turns_dnd(instances: list[dict]) -> list[dict]:
    rows: list[dict] = []

    for inst in instances:
        inp = inst["input"]
        counts_l, values_l = inp["count"], inp["value"]
        counts = {"book": counts_l[0], "hat": counts_l[1], "ball": counts_l[2]}
        vals = {"book": values_l[0], "hat": values_l[1], "ball": values_l[2]}
        max_pts = sum(counts[i] * vals[i] for i in DND_ITEMS)
        if max_pts == 0:
            continue

        system = _DND_SYSTEM.format(
            n_book=counts["book"], n_hat=counts["hat"], n_ball=counts["ball"],
            book_pts=vals["book"], hat_pts=vals["hat"], ball_pts=vals["ball"],
            max_pts=max_pts,
        )

        dialogue_str = inst.get("dialogue", "")
        turns = _parse_dnd_dialogue(dialogue_str)
        you_alloc, _ = _parse_dnd_output(inst.get("output", ""))

        has_prior = False
        for idx, turn in enumerate(turns):
            if turn["id"] != "YOU":
                continue
            if idx < 1:
                continue

            text = turn["text"]
            parsed = _parse_quantities(text, DND_ITEMS, counts)
            atype = _action_type(text, has_prior)

            if parsed is not None:
                alloc = {i: parsed.get(i, 0) for i in DND_ITEMS}
                has_prior = True
            elif atype == "accept" and you_alloc:
                alloc = dict(you_alloc)
            elif atype in ("accept", "reject"):
                alloc = {}
            else:
                continue

            action = {"type": atype, **alloc} if alloc else {"type": atype}
            hist = _fmt_history_simple(turns, "YOU", idx)
            thought = _det_thought_multi(
                vals, counts,
                alloc or {i: 0 for i in DND_ITEMS},
                DND_ITEMS, "unknown", atype,
            )

            rows.append({
                "task": "turn_dnd",
                "prompt": system + f"\n\nDialogue so far:\n{hist}" + _TURN_INSTRUCTION,
                "talk": text,
                "action": action,
                "det_thought": thought,
                "strategy_label": None,
            })

    return rows


_PRICE_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")


def _interleave_cl(
    agent_texts: list[str],
    opp_texts: list[str],
    agent_role: str,
    opp_role: str,
) -> list[dict]:
    """Reconstruct chronological turn order from separate agent/opponent lists."""
    first_role, first = opp_role, opp_texts
    second_role, second = agent_role, agent_texts

    turns: list[dict] = []
    for i in range(max(len(first), len(second))):
        if i < len(first):
            turns.append({"role": first_role, "text": first[i]})
        if i < len(second):
            turns.append({"role": second_role, "text": second[i]})
    return turns


def _strip_role_prefix(line: str) -> str:
    colon = line.find(":")
    return line[colon + 1:].strip() if colon > 0 else line.strip()


def generate_turns_cl(instances: list[dict]) -> list[dict]:
    rows: list[dict] = []

    for inst in instances:
        meta = inst.get("metadata", {})
        if not isinstance(meta, dict):
            meta = json.loads(meta) if isinstance(meta, str) else {}

        parsed_info = inst.get("_parsed", {})
        agent_role = parsed_info.get("role", meta.get("perspective", "buyer"))
        opp_role = "seller" if agent_role == "buyer" else "buyer"
        listing_price = parsed_info.get("listing_price", 0.0)
        product = parsed_info.get("product_name", "unknown")
        category = parsed_info.get("category", "unknown")
        successful = meta.get("successful", False)

        if listing_price <= 0:
            continue

        system = _CL_SYSTEM.format(
            role=agent_role, product_name=product,
            listing_price=listing_price, category=category,
        )

        agent_raw = [ln.strip() for ln in inst.get("output", "").split("\n") if ln.strip()]
        opp_raw = [ln.strip() for ln in inst.get("input", "").split("\n") if ln.strip()]
        agent_texts = [_strip_role_prefix(ln) for ln in agent_raw]
        opp_texts = [_strip_role_prefix(ln) for ln in opp_raw]
        chrono = _interleave_cl(agent_texts, opp_texts, agent_role, opp_role)

        for idx, turn in enumerate(chrono):
            if turn["role"] != agent_role:
                continue
            if idx < 1:
                continue

            text = turn["text"]
            is_last = idx == len(chrono) - 1
            atype = infer_action(text, is_last, successful)

            prices = extract_prices(text)
            price = prices[-1] if prices else None

            if atype in ("accept", "reject"):
                action: dict = {"type": atype}
            elif price is not None:
                action = {"type": atype, "price": price}
            else:
                continue

            hist_lines = [
                f"{'You' if t['role'] == agent_role else 'Partner'}: {t['text']}"
                for t in chrono[:idx]
            ]
            hist = "\n".join(hist_lines) or "(no dialogue yet)"

            if price is not None:
                pct = price / listing_price
                thought = (
                    f"Listing price: ${listing_price:.2f}. "
                    f"My proposed price: ${price:.2f} ({pct:.0%} of listing). "
                    f"Decision: {atype}."
                )
            else:
                thought = f"Decision: {atype}."

            rows.append({
                "task": "turn_cl",
                "prompt": system + f"\n\nDialogue so far:\n{hist}" + _CL_TURN_INSTRUCTION,
                "talk": text,
                "action": action,
                "det_thought": thought,
                "strategy_label": None,
            })

    return rows


TURN_GENERATORS: dict[str, type] = {
    "turn_ca": generate_turns_ca,
    "turn_dnd": generate_turns_dnd,
    "turn_cl": generate_turns_cl,
}
