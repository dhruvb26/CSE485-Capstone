import re
from html import escape

import streamlit as st

from rl.config import load_generate_config
from rl.streamlit.data_loader import count_rows, load_annotations, load_single

CONFIG_PATH = "rl/configs/generate.yaml"

_TAG_RE = re.compile(r"<(thought|talk|action)>(.*?)</\1>", re.DOTALL)


def _render_turns(chat_logs: list[dict], selected_agent: str):
    """Render raw conversation turns using the same ann-card layout."""
    for turn in chat_logs:
        text = turn["text"]
        is_me = turn["id"] == selected_agent
        card_cls = "ann-assistant" if is_me else "ann-user"
        role = "you" if is_me else "neighbor"
        st.markdown(
            f'<div class="ann-card {card_cls}">'
            f'<div class="ann-role">{role}</div>'
            f'<div class="ann-body">{escape(text)}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


def _parse_tagged_content(content: str) -> list[tuple[str, str]]:
    """Extract (tag_name, text) pairs from tagged assistant content."""
    parts = []
    for m in _TAG_RE.finditer(content):
        tag, text = m.group(1), m.group(2).strip()
        if text:
            parts.append((tag, text))
    return parts


def _render_annotated_turns(messages: list[dict]):
    """Render annotated messages with parsed thought/talk/action sections."""
    for i, message in enumerate(messages):
        role = message["role"]
        content = message["content"]

        if role == "system":
            st.markdown(
                f'<div class="ann-card ann-system">'
                f"<details><summary class=\"ann-role\">system</summary>"
                f'<div class="ann-body">{escape(content)}</div>'
                f"</details></div>",
                unsafe_allow_html=True,
            )
        elif role == "user":
            parts = _parse_tagged_content(content)
            if parts:
                inner = ""
                for tag, text in parts:
                    label = {"thought": "Thought", "talk": "Talk", "action": "Action"}[tag]
                    inner += (
                        f'<div class="ann-section">'
                        f'<span class="ann-section-label">{label}</span> '
                        f'<span class="ann-section-text">{escape(text)}</span>'
                        f"</div>"
                    )
                st.markdown(
                    f'<div class="ann-card ann-user">'
                    f'<div class="ann-role">neighbor</div>'
                    f"{inner}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="ann-card ann-user">'
                    f'<div class="ann-role">neighbor</div>'
                    f'<div class="ann-body">{escape(content)}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
        elif role == "assistant":
            parts = _parse_tagged_content(content)
            if not parts:
                st.markdown(
                    f'<div class="ann-card ann-assistant">'
                    f'<div class="ann-role">you (assistant)</div>'
                    f'<div class="ann-body">{escape(content)}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                continue

            inner = ""
            for tag, text in parts:
                label = {"thought": "Thought", "talk": "Talk", "action": "Action"}[tag]
                inner += (
                    f'<div class="ann-section">'
                    f'<span class="ann-section-label">{label}</span> '
                    f'<span class="ann-section-text">{escape(text)}</span>'
                    f"</div>"
                )

            st.markdown(
                f'<div class="ann-card ann-assistant">'
                f'<div class="ann-role">you (assistant)</div>'
                f"{inner}</div>",
                unsafe_allow_html=True,
            )


def _render_participant_info(participant_info: dict, selected_agent: str):
    """Render participant info for the selected agent."""
    st.json(participant_info[selected_agent], expanded=True)


def render():
    """Main entry point for the Dataset Viewer view."""
    try:
        config = load_generate_config(CONFIG_PATH)
    except Exception as e:
        st.error(f"Failed to load config from {CONFIG_PATH}: {e}")
        return

    annotations = load_annotations(config.output_jsonl)

    st.sidebar.markdown("## Configuration")

    ds_names = list(config.datasets.keys())
    selected_ds = st.sidebar.radio("Dataset", ds_names)
    ds_cfg = config.datasets[selected_ds]

    try:
        total = count_rows(ds_cfg.path)
    except Exception as e:
        st.error(f"Failed to read {ds_cfg.path}: {e}")
        return

    row_idx = st.sidebar.number_input(
        "Line Number",
        min_value=0,
        max_value=max(total - 1, 0),
        value=0,
        step=1,
    )

    st.sidebar.markdown(f"**{total}** conversations in `{selected_ds}`")

    try:
        chat_logs, participant_info, agent_ids = load_single(ds_cfg.path, row_idx)
    except Exception as e:
        st.error(f"Failed to load row {row_idx}: {e}")
        return

    badge_slot = st.empty()

    control_cols = st.columns(2, gap="large")
    with control_cols[0]:
        view_mode = st.selectbox("Conversation View", ["Raw", "Annotated"])
    with control_cols[1]:
        selected_agent = st.selectbox("Perspective", agent_ids)

    badge_slot.markdown(f"### `{selected_agent}`")

    annotation_key = f"{selected_ds}_{row_idx}_{selected_agent}"
    left, right = st.columns(2, gap="large")

    with left:
        if view_mode == "Annotated":
            if annotation_key in annotations:
                _render_annotated_turns(annotations[annotation_key])
            else:
                st.info("No annotation found for this conversation and agent.")
        else:
            _render_turns(chat_logs, selected_agent)

    with right:
        _render_participant_info(participant_info, selected_agent)
