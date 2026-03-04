"""Format validity verifier: checks that raw model output uses the XML-delimited
thought/talk/action format with a valid JSON action block."""

from __future__ import annotations

import json
import re

_ACTION_TYPES = {"offer", "counter", "accept", "reject", "propose"}

_THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL | re.IGNORECASE)
_TALK_RE = re.compile(r"<talk>(.*?)</talk>", re.DOTALL | re.IGNORECASE)
_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)


def check_format(raw_text: str) -> tuple[bool, str]:
    """Validate that *raw_text* contains ``<thought>``, ``<talk>``, and
    ``<action>`` XML tags, and that the action content is valid JSON with
    a recognised ``type`` field.

    Returns ``(valid, detail)`` where *detail* explains the failure.
    """
    parsed = parse_xml_turn(raw_text)
    if parsed is None:
        return False, "Output does not contain valid <thought>, <talk>, <action> XML tags."

    action_str = parsed["action_raw"]
    try:
        action = json.loads(action_str)
    except json.JSONDecodeError:
        return False, "Content inside <action> is not valid JSON."

    if not isinstance(action, dict):
        return False, "Action JSON is not a dict."

    action_type = action.get("type")
    if action_type not in _ACTION_TYPES:
        return False, f"action.type '{action_type}' not in {_ACTION_TYPES}"

    return True, "ok"


def parse_xml_turn(text: str) -> dict | None:
    """Extract thought/talk/action from XML-tagged output.

    Returns a dict with keys ``thought``, ``talk``, ``action_raw`` (unparsed
    JSON string), or ``None`` if any tag is missing.
    """
    thought_m = _THOUGHT_RE.search(text)
    talk_m = _TALK_RE.search(text)
    action_m = _ACTION_RE.search(text)

    if thought_m and talk_m and action_m:
        return {
            "thought": thought_m.group(1).strip(),
            "talk": talk_m.group(1).strip(),
            "action_raw": action_m.group(1).strip(),
        }

    return None
