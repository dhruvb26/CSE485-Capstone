import csv
import json
import logging
from itertools import islice

log = logging.getLogger(__name__)

POINTS = {"High": 5, "Medium": 4, "Low": 3}
STRUCTURAL = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}
ACTION_MAP = {
    "Accept-Deal": "[ACCEPT_DEAL]",
    "Reject-Deal": "[REJECT_DEAL]",
    "Walk-Away": "[WALK_AWAY]",
}


def build_system_prompt(participant_info: dict, agent_id: str) -> str:
    """Construct the negotiation system prompt for a given agent.

    Uses the agent's priority mapping and point values to produce a
    fully-templated system message with examples.
    """
    value2issue = participant_info[agent_id]["value2issue"]
    value2reason = participant_info[agent_id]["value2reason"]
    priorities = [
        (lvl, value2issue[lvl], POINTS[lvl]) for lvl in ("High", "Medium", "Low")
    ]
    high_issue, med_issue, low_issue = [p[1] for p in priorities]
    high_pts, med_pts, low_pts = [p[2] for p in priorities]

    priorities_str = "\n".join(
        f"- {issue} (3 available): {lvl} priority ({pts} pts each) - Reason: {value2reason.get(lvl, 'N/A')}"
        for lvl, issue, pts in priorities
    )

    example_alloc = {high_issue.lower(): 3, med_issue.lower(): 2, low_issue.lower(): 1}
    example_pts = 3 * high_pts + 2 * med_pts + 1 * low_pts
    example_action = (
        f"[SUBMIT_DEAL] "
        f"food:{example_alloc.get('food', 0)} "
        f"water:{example_alloc.get('water', 0)} "
        f"firewood:{example_alloc.get('firewood', 0)}"
    )

    return f"""\
You are negotiating with your campsite neighbor over extra supply of food, water, and firewood for your camping trip.

There are exactly 3 packages of each item (food, water, firewood) to divide between you and your neighbor. Each item allocation in a deal must be between 0 and 3, and the two parties' allocations for each item must sum to 3.

Your item priorities and point values:
{priorities_str}

Your reply must always include all 3 parts in this order:

<thought>your inner strategic thinking of this bargaining session.</thought>
<talk>short talk that you are going to say to the neighbor. Speak concisely and cut to the chase. Generate authentic and diverse sentences, avoiding repetition of sentences that have already appeared in the conversation.</talk>
<action>one of: [TALK] | [SUBMIT_DEAL] food:F water:W firewood:FW | [ACCEPT_DEAL] | [REJECT_DEAL] | [WALK_AWAY]</action>

Your neighbor's messages show their <talk> and <action> (or just <action> for structural moves like deals).

For example:

<thought>They haven't proposed yet. My priorities are {high_issue} > {med_issue} > {low_issue}. I'll probe their needs before anchoring.</thought>
<talk>Hi! Happy to work something out. What are you most in need of for your trip?</talk>
<action>[TALK]</action>

Or when proposing a deal:

<thought>{high_issue} is my top priority at {high_pts}pts each. Anchoring at 3 {high_issue} + 2 {med_issue} + 1 {low_issue} = {3 * high_pts}+{2 * med_pts}+{1 * low_pts}={example_pts}pts.</thought>
<talk>How about I take 3 {high_issue.lower()}, 2 {med_issue.lower()}, and 1 {low_issue.lower()} - and you take the rest?</talk>
<action>{example_action}</action>"""


def render_turn(turn: dict, agent_id: str) -> tuple[str, str, str]:
    """Render a single conversation turn into tagged components.

    Returns a tuple of (talk_part, action_part, rendered) where rendered
    is the combined string used for conversation history display.
    """
    text = turn["text"]
    td = turn["task_data"]
    is_me = turn["id"] == agent_id

    if text == "Submit-Deal":
        alloc = td["issue2youget"] if is_me else td["issue2theyget"]
        action_str = (
            f"[SUBMIT_DEAL] "
            f"food:{alloc.get('Food', 0)} "
            f"water:{alloc.get('Water', 0)} "
            f"firewood:{alloc.get('Firewood', 0)}"
        )
        return (
            "<talk></talk>\n",
            f"<action>{action_str}</action>",
            f"<action>{action_str}</action>",
        )

    if text in ACTION_MAP:
        ap = f"<action>{ACTION_MAP[text]}</action>"
        return "<talk></talk>\n", ap, ap

    return (
        f"<talk>{text}</talk>\n",
        "<action>[TALK]</action>",
        f"<talk>{text}</talk> <action>[TALK]</action>",
    )


def parse_tag_content(text: str, open_tag: str, close_tag: str) -> str | None:
    """Extract text between an open and close tag pair.

    Returns the stripped content between the tags, or None if either
    tag is not found.
    """
    start = text.find(open_tag)
    if start == -1:
        return None
    start += len(open_tag)
    end = text.find(close_tag, start)
    if end == -1:
        return None
    return text[start:end].strip()


def _merge_assistant_contents(prev: str, new: str) -> str:
    """Collapse two consecutive assistant contents into one.

    Combines thoughts, prefers the non-empty talk, and keeps the
    structural action (anything other than [TALK]) over [TALK].
    """
    prev_thought = parse_tag_content(prev, "<thought>", "</thought>") or ""
    new_thought = parse_tag_content(new, "<thought>", "</thought>") or ""
    combined_thought = f"{prev_thought} {new_thought}".strip()

    prev_talk = parse_tag_content(prev, "<talk>", "</talk>") or ""
    new_talk = parse_tag_content(new, "<talk>", "</talk>") or ""
    combined_talk = new_talk or prev_talk

    prev_action = parse_tag_content(prev, "<action>", "</action>") or ""
    new_action = parse_tag_content(new, "<action>", "</action>") or ""
    structural = next(
        (a for a in (prev_action, new_action) if a != "[TALK]"), prev_action
    )
    combined_action = structural

    return (
        f"<thought>{combined_thought}</thought>\n"
        f"<talk>{combined_talk}</talk>\n"
        f"<action>{combined_action}</action>"
    )


def _append_assistant(messages: list[dict], content: str):
    """Append an assistant turn, handling kickoff and consecutive-turn merging."""
    if messages[-1]["role"] == "system":
        messages.append({"role": "user", "content": "Begin the negotiation."})

    if messages[-1]["role"] == "assistant":
        messages[-1]["content"] = _merge_assistant_contents(
            messages[-1]["content"],
            content,
        )
    else:
        messages.append({"role": "assistant", "content": content})


def build_sft_messages(
    chat_logs: list[dict], participant_info: dict, agent_id: str
) -> list[dict]:
    """Build a complete SFT-ready messages list from raw chat logs.

    Converts the full conversation into an OpenAI-style messages list
    (system / user / assistant turns) suitable for apply_chat_template.
    """
    messages = [
        {"role": "system", "content": build_system_prompt(participant_info, agent_id)}
    ]
    pending = []

    for turn in chat_logs:
        is_me = turn["id"] == agent_id
        talk_part, action_part, rendered = render_turn(turn, agent_id)

        if is_me:
            if pending:
                messages.append({"role": "user", "content": "\n".join(pending)})
                pending = []
            _append_assistant(
                messages, f"<thought></thought>\n{talk_part}{action_part}"
            )
        else:
            pending.append(f"{rendered}")

    if pending:
        messages.append({"role": "user", "content": "\n".join(pending)})

    return messages


def build_annotation_context(participant_info: dict, agent_id: str) -> tuple[str, str]:
    """Return the (gpt_system_prompt, priorities_str) used for annotation requests."""
    value2issue = participant_info[agent_id]["value2issue"]
    priorities = [
        (lvl, value2issue[lvl], POINTS[lvl]) for lvl in ("High", "Medium", "Low")
    ]

    priorities_str = "\n".join(
        f"- {issue} (3 available): {lvl} priority ({pts} pts each)"
        for lvl, issue, pts in priorities
    )

    gpt_system = """\
    You are a negotiation analyst. Given a camping-supply negotiation, generate a realistic <thought> monologue for the specified agent at a specific turn.

    There are exactly 3 packages of each item (food, water, firewood) to divide between the two parties. Each item allocation must be between 0 and 3, and the two parties' allocations for each item must sum to 3.

    Rules for the <thought>:
    - When evaluating or proposing a deal, include explicit point arithmetic (e.g. '3 food x 5 pts = 15').
    - For conversational turns, focus on strategic reasoning and modeling the partner's likely priorities.
    - State the strategic rationale for the action that was actually taken.
    - 2-4 sentences.

    If the turn's action is structural (Submit-Deal, Accept-Deal, Reject-Deal, Walk-Away) also generate a short <talk> line the agent might say alongside that action.
    Otherwise, output ONLY <thought>...</thought> (no <talk>)."""

    return gpt_system, priorities_str


def build_annotation_requests(
    chat_logs: list[dict], participant_info: dict, agent_id: str
) -> list[dict]:
    """Build per-turn GPT annotation request payloads for a given agent.

    Each returned dict contains turn_idx, needs_talk (bool), and messages
    (a system+user message list ready for the OpenAI chat completions API).

    NOTE: These requests are independent — they do NOT include prior generated
    thoughts.  For sequential annotation with thought accumulation, use
    ``generate.annotate_agent`` directly.
    """
    gpt_system, priorities_str = build_annotation_context(participant_info, agent_id)

    requests = []
    history_lines: list[str] = []

    for idx, turn in enumerate(chat_logs):
        is_me = turn["id"] == agent_id
        text = turn["text"]
        _, _, rendered = render_turn(turn, agent_id)

        if not is_me:
            history_lines.append(f"Them: {rendered}")
            continue

        is_structural = text in STRUCTURAL
        history_block = (
            "\n".join(history_lines)
            if history_lines
            else "(opening turn - no prior messages)"
        )

        user_prompt_parts = [
            f"Agent priorities:\n{priorities_str}\n",
            f"Conversation so far:\n{history_block}\n",
            f"Action taken this turn: {rendered}",
        ]
        if not is_structural:
            user_prompt_parts.append(f'Talk text this turn: "{text}"')

        if is_structural:
            user_prompt_parts.append(
                "\nThis is a structural action with no human text. "
                "Generate both <thought>...</thought> and <talk>...</talk>."
            )
        else:
            user_prompt_parts.append(
                "\nGenerate only <thought>...</thought> for this turn."
            )

        requests.append(
            {
                "turn_idx": idx,
                "needs_talk": is_structural,
                "messages": [
                    {"role": "system", "content": gpt_system},
                    {"role": "user", "content": "\n".join(user_prompt_parts)},
                ],
            }
        )

        history_lines.append(f"You: {rendered}")

    return requests


def merge_annotations(
    chat_logs: list[dict],
    participant_info: dict,
    agent_id: str,
    annotations: dict[int, dict],
) -> list[dict]:
    """Build the final SFT messages list with <thought> (and optionally <talk>) filled from annotations."""

    messages = [
        {"role": "system", "content": build_system_prompt(participant_info, agent_id)}
    ]
    pending: list[str] = []

    for idx, turn in enumerate(chat_logs):
        is_me = turn["id"] == agent_id
        talk_part, action_part, rendered = render_turn(turn, agent_id)

        if is_me:
            if pending:
                messages.append({"role": "user", "content": "\n".join(pending)})
                pending = []

            ann = annotations.get(idx, {})
            thought = ann.get("thought", "")
            thought_str = f"<thought>{thought}</thought>"

            if ann.get("talk") is not None:
                talk_part = f"<talk>{ann['talk']}</talk>\n"

            _append_assistant(messages, f"{thought_str}\n{talk_part}{action_part}")
        else:
            pending.append(f"{rendered}")

    if pending:
        messages.append({"role": "user", "content": "\n".join(pending)})

    return messages


def count_conversations(csv_path: str) -> int:
    """Return the number of data rows in a CaSiNo CSV (excludes header)."""
    try:
        with open(csv_path, "r") as f:
            return sum(1 for _ in f) - 1
    except FileNotFoundError:
        log.error("CSV file not found: %s", csv_path)
        raise


def load_conversation(
    csv_path: str, row_idx: int
) -> tuple[list[dict], dict, list[str]]:
    """Load a single conversation from the CaSiNo CSV dataset.

    Returns (chat_logs, participant_info, agent_ids) for the row at
    the given index.
    """
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            rows = list(islice(reader, row_idx, row_idx + 1))
            if not rows:
                raise IndexError(f"Row {row_idx} not found in {csv_path}")
            row = rows[0]
    except FileNotFoundError:
        log.error("CSV file not found: %s", csv_path)
        raise
    except Exception:
        log.error("Failed to read conversation from %s at row %d", csv_path, row_idx)
        raise

    chat_logs = json.loads(row[0])
    participant_info = json.loads(row[1])
    agent_ids = list(participant_info.keys())
    return chat_logs, participant_info, agent_ids


def load_all_conversations(
    csv_path: str,
) -> list[tuple[int, list[dict], dict, list[str]]]:
    """Load every conversation from a CaSiNo CSV dataset.

    Returns a list of (row_idx, chat_logs, participant_info, agent_ids)
    tuples, one per row in the file.
    """
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            rows = list(reader)
    except FileNotFoundError:
        log.error("CSV file not found: %s", csv_path)
        raise
    except Exception:
        log.error("Failed to read conversations from %s", csv_path)
        raise

    conversations = []
    for idx, row in enumerate(rows):
        chat_logs = json.loads(row[0])
        participant_info = json.loads(row[1])
        agent_ids = list(participant_info.keys())
        conversations.append((idx, chat_logs, participant_info, agent_ids))

    return conversations
