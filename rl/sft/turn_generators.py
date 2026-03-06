"""Turn-level SFT generators for CaSiNo and DND.

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

import logging
import re

from rl.handlers.casino.dataset import (
    PRIORITY_TO_POINTS,
    STRATEGY_LABEL_MAP,
    STRATEGY_LABELS,
    _META_TURNS,
    sanitize_unicode,
)

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

_TURN_INSTRUCTION = (
    "\n\nProduce your next negotiation turn using the following XML format:\n"
    "<thought>your internal reasoning (point arithmetic, partner priority estimate, justification)</thought>\n"
    "<talk>your natural language response to your partner</talk>\n"
    '<action>{"type": "offer"|"counter"|"accept"|"reject", ...item allocations for yourself}</action>'
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

                prior_real = [t for t in chat_logs[:idx] if t["text"] not in _META_TURNS]

                text = sanitize_unicode(turn["text"])
                parsed = _parse_quantities(text, CA_ITEMS, CA_COUNTS)
                atype = _action_type(text, has_prior)

                if not prior_real:
                    # Turn-0: opening move with no dialogue history
                    if parsed is not None:
                        alloc = {i: parsed.get(i, 0) for i in CA_ITEMS}
                    else:
                        best_item = max(CA_ITEMS, key=lambda i: vals[i])
                        alloc = {i: (CA_COUNTS[i] if i == best_item else 1) for i in CA_ITEMS}
                    atype = "offer"
                    has_prior = True
                    action = {"type": atype, **alloc}
                    ppri = "unknown"
                    thought = _det_thought_multi(vals, CA_COUNTS, alloc, CA_ITEMS, ppri, atype)
                    rows.append({
                        "task": "turn_ca",
                        "prompt": system + _TURN_INSTRUCTION,
                        "talk": text,
                        "action": action,
                        "det_thought": thought,
                        "strategy_label": None,
                    })
                    continue

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
    """Parse 'item0=X item1=Y ...' → (you_alloc, them_alloc).

    Returns empty dicts for no-agreement / disagree outcomes.
    """
    stripped = output_str.strip()
    if "<no_agreement>" in stripped or "<disagree>" in stripped:
        return {}, {}
    vals: list[int] = []
    for tok in stripped.split():
        if "=" in tok:
            vals.append(int(tok.split("=")[1]))
    if len(vals) < 6:
        logger.debug("Malformed DND output (expected 6 values, got %d): %s", len(vals), stripped[:100])
    you = {"book": vals[0], "hat": vals[1], "ball": vals[2]} if len(vals) >= 3 else {}
    them = {"book": vals[3], "hat": vals[4], "ball": vals[5]} if len(vals) >= 6 else {}
    return you, them


_DND_TO_CA = {"book": "food", "hat": "water", "ball": "firewood"}


def _remap_dnd_alloc(alloc: dict[str, int]) -> dict[str, int]:
    return {_DND_TO_CA[k]: v for k, v in alloc.items()}


def generate_turns_dnd(instances: list[dict]) -> list[dict]:
    rows: list[dict] = []

    for inst in instances:
        inp = inst["input"]
        counts_l, values_l = inp["count"], inp["value"]
        raw_counts = {"book": counts_l[0], "hat": counts_l[1], "ball": counts_l[2]}
        raw_vals = {"book": values_l[0], "hat": values_l[1], "ball": values_l[2]}

        counts = _remap_dnd_alloc(raw_counts)
        vals = _remap_dnd_alloc(raw_vals)
        max_pts = sum(counts[i] * vals[i] for i in CA_ITEMS)
        if max_pts == 0:
            continue

        system = _CA_SYSTEM.format(
            food=vals["food"], water=vals["water"],
            firewood=vals["firewood"], max_pts=max_pts,
        )

        dialogue_str = inst.get("dialogue", "")
        turns = _parse_dnd_dialogue(dialogue_str)
        raw_you_alloc, _ = _parse_dnd_output(inst.get("output", ""))
        you_alloc = _remap_dnd_alloc(raw_you_alloc) if raw_you_alloc else {}

        has_prior = False
        for idx, turn in enumerate(turns):
            if turn["id"] != "YOU":
                continue
            if idx < 1:
                continue

            text = turn["text"]
            parsed = _parse_quantities(text, DND_ITEMS, raw_counts)
            atype = _action_type(text, has_prior)

            if parsed is not None:
                alloc = _remap_dnd_alloc({i: parsed.get(i, 0) for i in DND_ITEMS})
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
                alloc or {i: 0 for i in CA_ITEMS},
                CA_ITEMS, "unknown", atype,
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


TURN_GENERATORS: dict[str, type] = {
    "turn_ca": generate_turns_ca,
    "turn_dnd": generate_turns_dnd,
}
