"""Render episodes.json to a simple, readable HTML file."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path


def _extract_section(text: str, section: str) -> str | None:
    pat = re.compile(
        rf"(?:^|\n)\s*{section}\s*:\s*(.*?)(?=\n\s*(?:Thought|Talk|Action)\s*:|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    m = pat.search(text)
    return m.group(1).strip() if m else None


def _parse_message(content: str) -> dict:
    """Parse a model output into Thought/Talk/Action parts.

    Returns a dict with 'thought', 'talk', 'action' keys (None if missing),
    plus 'malformed' bool indicating parse failure.
    """
    thought = _extract_section(content, "Thought")
    talk = _extract_section(content, "Talk")
    action = _extract_section(content, "Action")
    malformed = thought is None or talk is None or action is None
    return {
        "thought": thought,
        "talk": talk,
        "action": action,
        "malformed": malformed,
    }


def _esc(text: str | None) -> str:
    if text is None:
        return ""
    return html.escape(text)


def _render_episode(ep: dict, idx: int) -> str:
    outcome = ep.get("outcome", "unknown")
    outcome_cls = {
        "deal": "tag-deal",
        "walk_away": "tag-walkaway",
        "reject_loop": "tag-reject",
        "max_turns": "tag-maxturns",
    }.get(outcome, "tag-unknown")

    learner_ok = ep.get("learner_format_ok", 0)
    learner_total = ep.get("learner_total_turns", 0)
    opponent_ok = ep.get("opponent_format_ok", 0)
    opponent_total = ep.get("opponent_total_turns", 0)

    pts = ""
    if outcome == "deal":
        pts = (
            f'<span class="meta-sep">/</span>'
            f'<span class="meta-label">Points</span> '
            f"L {ep.get('learner_points')}  O {ep.get('opponent_points')}"
        )

    parts = [f"<details {'open' if idx == 0 else ''}>"]
    parts.append(
        f"<summary>"
        f'<span class="ep-num">#{ep["episode_id"]}</span>'
        f'<span class="{outcome_cls}">{outcome}</span>'
        f'<span class="ep-turns">{ep.get("num_turns", 0)} turns</span>'
        f"</summary>"
    )

    parts.append('<div class="ep-meta">')
    parts.append(
        f'<span class="meta-label">Persona</span> {_esc(ep.get("persona", "none"))}'
        f'<span class="meta-sep">/</span>'
        f'<span class="meta-label">Terminated by</span> {_esc(ep.get("who_terminated", ""))}'
        f'<span class="meta-sep">/</span>'
        f'<span class="meta-label">Format</span> '
        f"L {learner_ok}/{learner_total}  O {opponent_ok}/{opponent_total}"
        f"{pts}"
    )
    parts.append("</div>")

    learner_msgs = ep.get("learner_messages", [])
    opponent_msgs = ep.get("opponent_messages", [])
    dialogue = _interleave_messages(learner_msgs, opponent_msgs)

    for speaker, role, msg in dialogue:
        content = msg["content"]

        if role == "system":
            parts.append('<div class="msg msg-system">')
            parts.append(
                f'<div class="msg-header">System <span class="role-sub">({speaker})</span></div>'
            )
            truncated = content[:300] + ("..." if len(content) > 300 else "")
            parts.append(f"<pre>{_esc(truncated)}</pre>")
            parts.append("</div>")
            continue

        if role == "assistant":
            parsed = _parse_message(content)
            format_ok = msg.get("format_ok", True)
            css = "msg-learner" if speaker == "Learner" else "msg-opponent"
            parts.append(f'<div class="msg {css}">')
            parts.append(f'<div class="msg-header">{speaker}')
            if not format_ok:
                parts.append(' <span class="malformed-tag">malformed</span>')
            parts.append("</div>")

            for key, label in [("thought", "Thought"), ("talk", "Talk")]:
                if parsed[key]:
                    cls = key
                    parts.append(
                        f'<div class="field {cls}"><span class="field-label">{label}</span> {_esc(parsed[key])}</div>'
                    )
            if parsed["action"]:
                parts.append(
                    f'<div class="field action"><span class="field-label">Action</span> <code>{_esc(parsed["action"])}</code></div>'
                )

            if not format_ok:
                parts.append(
                    '<details class="raw-details"><summary class="raw-summary">Raw output</summary>'
                )
                parts.append(f'<pre class="raw-output">{_esc(content)}</pre>')
                parts.append("</details>")

            parts.append("</div>")

    parts.append("</details>")
    return "\n".join(parts)


def _interleave_messages(
    learner_msgs: list[dict], opponent_msgs: list[dict]
) -> list[tuple[str, str, dict]]:
    """Interleave learner and opponent messages into dialogue order.

    System messages come first, then alternating turns reconstructed from
    the user/assistant pairs in each message list.
    Returns (speaker, role, msg_dict) tuples.
    """
    dialogue: list[tuple[str, str, dict]] = []

    l_sys = [m for m in learner_msgs if m["role"] == "system"]
    o_sys = [m for m in opponent_msgs if m["role"] == "system"]
    if l_sys:
        dialogue.append(("Learner", "system", l_sys[0]))
    if o_sys:
        dialogue.append(("Opponent", "system", o_sys[0]))

    l_turns = [m for m in learner_msgs if m["role"] == "assistant"]
    o_turns = [m for m in opponent_msgs if m["role"] == "assistant"]

    li, oi = 0, 0
    opponent_goes_first = (
        len(o_turns) > 0
        and len(l_turns) > 0
        and len(learner_msgs) > 1
        and learner_msgs[1]["role"] == "user"
    )

    if opponent_goes_first:
        if oi < len(o_turns):
            dialogue.append(("Opponent", "assistant", o_turns[oi]))
            oi += 1

    while li < len(l_turns) or oi < len(o_turns):
        if li < len(l_turns):
            dialogue.append(("Learner", "assistant", l_turns[li]))
            li += 1
        if oi < len(o_turns):
            dialogue.append(("Opponent", "assistant", o_turns[oi]))
            oi += 1

    return dialogue


_CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 14px;
    line-height: 1.6;
    color: #09090b;
    background: #fafafa;
    max-width: 860px;
    margin: 0 auto;
    padding: 40px 24px 80px;
}

h1 {
    font-size: 16px;
    font-weight: 600;
    color: #09090b;
    letter-spacing: -0.01em;
}

.header {
    margin-bottom: 32px;
    padding-bottom: 16px;
    border-bottom: 1px solid #e4e4e7;
}

.header p {
    font-size: 13px;
    color: #71717a;
    margin-top: 4px;
}

details {
    margin-bottom: 6px;
    border: 1px solid #e4e4e7;
    border-radius: 6px;
    background: #fff;
    overflow: hidden;
}

details[open] {
    margin-bottom: 12px;
}

summary {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    cursor: pointer;
    font-size: 13px;
    color: #3f3f46;
    user-select: none;
    list-style: none;
}
summary::-webkit-details-marker { display: none; }
summary::before {
    content: "\\25B8";
    font-size: 16px;
    color: #a1a1aa;
    transition: transform 0.15s;
}
details[open] > summary::before {
    transform: rotate(90deg);
}
summary:hover { background: #fafafa; }

.ep-num {
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    color: #09090b;
    min-width: 36px;
}

.ep-turns {
    margin-left: auto;
    font-size: 12px;
    color: #a1a1aa;
    font-variant-numeric: tabular-nums;
}

.tag-deal, .tag-walkaway, .tag-reject, .tag-maxturns, .tag-unknown {
    font-size: 11px;
    font-weight: 500;
    padding: 1px 8px;
    border-radius: 9999px;
}
.tag-deal       { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
.tag-walkaway   { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
.tag-reject     { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
.tag-maxturns   { background: #f4f4f5; color: #52525b; border: 1px solid #e4e4e7; }
.tag-unknown    { background: #f4f4f5; color: #71717a; border: 1px solid #e4e4e7; }

.ep-meta {
    padding: 8px 14px;
    font-size: 12px;
    color: #71717a;
    background: #fafafa;
    border-top: 1px solid #f4f4f5;
    border-bottom: 1px solid #f4f4f5;
    line-height: 1.8;
}
.meta-label {
    font-weight: 500;
    color: #52525b;
}
.meta-sep {
    margin: 0 8px;
    color: #d4d4d8;
}

.msg {
    padding: 10px 14px;
    border-top: 1px solid #f4f4f5;
}

.msg-header {
    font-size: 12px;
    font-weight: 600;
    color: #09090b;
    margin-bottom: 6px;
}
.role-sub {
    font-weight: 400;
    color: #a1a1aa;
}

.msg-system {
    background: #fafafa;
}
.msg-system pre {
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 12px;
    color: #71717a;
    max-height: 100px;
    overflow: auto;
    line-height: 1.5;
}

.msg-learner { background: #fafbff; }
.msg-opponent { background: #fffbfa; }

.field {
    margin-bottom: 4px;
    font-size: 13px;
    line-height: 1.55;
}
.field-label {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #a1a1aa;
    width: 60px;
    vertical-align: top;
}

.thought { color: #71717a; }
.talk { color: #27272a; }
.action code {
    font-family: "SF Mono", "Cascadia Code", "Fira Code", monospace;
    font-size: 12px;
    color: #3f3f46;
    background: #f4f4f5;
    padding: 1px 5px;
    border-radius: 3px;
}

.malformed-tag {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #b91c1c;
    background: #fef2f2;
    border: 1px solid #fecaca;
    padding: 1px 7px;
    border-radius: 9999px;
    margin-left: 6px;
    vertical-align: middle;
}

.raw-details {
    margin-top: 6px;
}
.raw-summary {
    font-size: 11px;
    color: #a1a1aa;
    cursor: pointer;
    user-select: none;
}
.raw-summary:hover { color: #71717a; }

.raw-output {
    white-space: pre-wrap;
    word-break: break-word;
    font-family: "SF Mono", "Cascadia Code", "Fira Code", monospace;
    font-size: 12px;
    line-height: 1.5;
    color: #3f3f46;
    background: #f4f4f5;
    border: 1px solid #e4e4e7;
    border-radius: 4px;
    padding: 8px 10px;
    max-height: 300px;
    overflow: auto;
}
"""


def render_episodes_html(
    episodes: list[dict],
    title: str = "Negotiation Episodes",
    dataset: str = "",
) -> str:
    subtitle = f"{len(episodes)} episodes"
    if dataset:
        subtitle = f"{dataset} / {subtitle}"
    parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{_esc(title)}</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        '<div class="header">',
        f"<h1>{_esc(title)}</h1>",
        f"<p>{_esc(subtitle)}</p>",
        "</div>",
    ]
    for i, ep in enumerate(episodes):
        parts.append(_render_episode(ep, i))
    parts.append("</body></html>")
    return "\n".join(parts)


def render_from_json(
    json_path: str | Path, output_path: str | Path | None = None
) -> Path:
    """Read episodes.json and write an HTML file next to it."""
    json_path = Path(json_path)
    with open(json_path) as f:
        episodes = json.load(f)

    matchup_dir = json_path.parent
    title = matchup_dir.name
    dataset = matchup_dir.parent.name if matchup_dir.parent.name != "negotiate" else ""
    html_content = render_episodes_html(episodes, title=title, dataset=dataset)

    if output_path is None:
        output_path = json_path.parent / "episodes.html"
    else:
        output_path = Path(output_path)

    with open(output_path, "w") as f:
        f.write(html_content)
    return output_path
