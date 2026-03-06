import json
import re
from pathlib import Path

import streamlit as st

RUNS_DIR = Path("runs")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 3.5rem;
                padding-bottom: 2rem;
            }
            div[data-testid="stMetric"] {
                background: transparent;
                border: none;
                padding: 0;
            }
            div[data-testid="stMetric"] label {
                font-size: 0.78rem;
                letter-spacing: 0.02em;
            }
            div[data-testid="stMetricValue"] {
                font-size: 1.45rem;
            }
            .eyebrow {
                font-size: 0.78rem;
                font-weight: 600;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                opacity: 0.72;
                margin-bottom: 0.35rem;
            }
            .page-title {
                font-size: 2.35rem;
                font-weight: 650;
                line-height: 1.05;
                margin-bottom: 0.35rem;
            }
            .page-subtitle {
                max-width: 52rem;
                opacity: 0.78;
                margin-bottom: 0;
            }
            .turn-talk {
                margin: 0.5rem 0 1rem;
                line-height: 1.65;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_episode(path: Path) -> tuple[list[dict], dict | None]:
    turns, summary = [], None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("episode_summary"):
            summary = rec
        else:
            turns.append(rec)
    return turns, summary


def extract_xml(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_system_prompt(prompt: str) -> str:
    """Pull the user message body from the ChatML prompt."""
    m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, re.DOTALL)
    return m.group(1).strip() if m else ""


def render_intro() -> None:
    st.markdown(
        '<div class="eyebrow">Training Run Explorer</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-title">GRPO Episode Viewer</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="page-subtitle">Inspect a single negotiation episode with cleaner structure,'
        " quick stats, scenario context, and turn-by-turn reasoning.</p>",
        unsafe_allow_html=True,
    )


def render_summary(summary: dict) -> None:
    cards = [
        ("Reward", f"{summary['reward']:.3f}", ":material/finance:"),
        ("Deal", "Yes" if summary["deal"] else "No", ":material/handshake:"),
        ("Turns", str(summary["turns"]), ":material/forum:"),
        ("Persona", summary["persona"], ":material/badge:"),
        ("Time", f"{summary['elapsed_s']:.1f}s", ":material/timer:"),
    ]
    cols = st.columns(len(cards))
    for col, (label, value, icon) in zip(cols, cards):
        with col:
            with st.container(border=True):
                st.metric(f"{icon} {label}", value)


def render_scenario(summary: dict | None, turns: list[dict]) -> None:
    if summary:
        sc = summary.get("scenario", {})
        items = sc.get("items", {})
        agent_vals = sc.get("agent_values", {})
        partner_vals = sc.get("partner_values", {})

        st.markdown("### :material/package_2: Scenario")
        col1, col2, col3 = st.columns(3)
        with col1:
            with st.container(border=True):
                st.markdown("##### :material/inventory_2: Items Available")
                for item, qty in items.items():
                    st.markdown(f"- `{item.title()}`: {qty}")
        with col2:
            with st.container(border=True):
                st.markdown("##### :material/school: Learner Values")
                for item, val in agent_vals.items():
                    st.markdown(f"- `{item.title()}`: {val} pts")
        with col3:
            with st.container(border=True):
                st.markdown("##### :material/groups_2: Partner Values")
                for item, val in partner_vals.items():
                    st.markdown(f"- `{item.title()}`: {val} pts")

    learner_prompt = next(
        (extract_system_prompt(t["prompt"]) for t in turns if t["agent"] == "learner"),
        None,
    )
    clone_prompt = next(
        (extract_system_prompt(t["prompt"]) for t in turns if t["agent"] == "clone"),
        None,
    )

    if learner_prompt or clone_prompt:
        st.markdown("### :material/terminal: System Prompts")
        labels, prompts = [], []
        if learner_prompt:
            labels.append(":material/school: Learner")
            prompts.append(learner_prompt)
        if clone_prompt:
            labels.append(":material/person: Clone")
            prompts.append(clone_prompt)

        for tab, prompt in zip(st.tabs(labels), prompts):
            with tab:
                st.code(prompt, language=None)


def render_turn(turn: dict) -> None:
    agent = turn["agent"]
    is_learner = agent == "learner"
    label = "Learner" if is_learner else "Clone"
    icon = ":material/school:" if is_learner else ":material/person:"

    response = turn.get("best_response") or turn.get("response", "")
    thought = extract_xml(response, "thought")
    talk = extract_xml(response, "talk")
    action = extract_xml(response, "action")

    head_col, _ = st.columns([1.3, 1], vertical_alignment="center")
    with head_col:
        st.markdown(f"#### {icon} {label}")

    with st.container(border=True):
        detail_cols = st.columns(2)
        with detail_cols[0]:
            if thought:
                with st.expander("Thought"):
                    st.markdown(thought)
        with detail_cols[1]:
            if action:
                with st.expander("Action"):
                    st.code(action, language="json")

        if talk:
            st.markdown(f'<div class="turn-talk">{talk}</div>', unsafe_allow_html=True)

        if is_learner and "scores" in turn:
            with st.expander("Candidates and Scores"):
                candidates = turn["candidates"]
                scores = turn["scores"]
                advantages = turn.get("advantages", [])
                best = turn.get("best_idx", -1)
                for i, (c, s) in enumerate(zip(candidates, scores)):
                    marker = " **best**" if i == best else ""
                    c_talk = extract_xml(c, "talk") or c[:80]
                    adv = f"  adv={advantages[i]:+.3f}" if i < len(advantages) else ""
                    st.markdown(f"`[{i}]` score={s:.3f}{adv}{marker}  \n> {c_talk}")

st.set_page_config(
    page_title="Episode Viewer", page_icon=":material/insights:", layout="wide"
)
inject_styles()

if not RUNS_DIR.exists():
    st.warning("No runs found in `runs/`.")
    st.stop()

run_dirs = sorted(
    [d for d in RUNS_DIR.iterdir() if d.is_dir() and (d / "episodes").exists()],
    reverse=True,
)

if not run_dirs:
    st.warning("No runs found in `runs/`.")
    st.stop()

with st.sidebar:
    st.markdown("### :material/folder_open: Browse")
    st.caption("Choose a run and episode to inspect.")
    run_name = st.selectbox("Run", [d.name for d in run_dirs])
    run_path = RUNS_DIR / run_name / "episodes"

    ep_files = sorted(run_path.glob("ep_*.jsonl"))
    if not ep_files:
        st.warning("No episode files found.")
        st.stop()

    ep_labels = [f.stem for f in ep_files]
    ep_choice = st.selectbox("Episode", ep_labels, index=len(ep_labels) - 1)
    st.divider()
    st.caption(f"{len(ep_files)} episodes in this run")

ep_path = run_path / f"{ep_choice}.jsonl"

turns, summary = load_episode(ep_path)
render_intro()

if summary:
    render_summary(summary)

render_scenario(summary, turns)

st.divider()
st.markdown("### :material/forum: Dialogue")

for turn in turns:
    render_turn(turn)
