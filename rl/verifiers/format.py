"""Format validity verifier: checks that raw model output parses as valid
thought/talk/action JSON with all required keys."""

from __future__ import annotations

import json
import re

_REQUIRED_KEYS = {"thought", "talk", "action"}
_ACTION_TYPES = {"offer", "counter", "accept", "reject"}


def check_format(raw_text: str) -> tuple[bool, str]:
    """Validate that *raw_text* is (or contains) a JSON object with the
    required ``thought``, ``talk``, and ``action`` fields.

    Returns ``(valid, detail)`` where *detail* explains the failure.
    """
    parsed = _try_parse(raw_text)
    if parsed is None:
        return False, "Output does not contain a valid JSON object."

    missing = _REQUIRED_KEYS - set(parsed.keys())
    if missing:
        return False, f"Missing required keys: {sorted(missing)}"

    action = parsed.get("action")
    if not isinstance(action, dict):
        return False, "action field is not a dict."

    action_type = action.get("type")
    if action_type not in _ACTION_TYPES:
        return False, f"action.type '{action_type}' not in {_ACTION_TYPES}"

    return True, "ok"


def _try_parse(text: str) -> dict | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None
