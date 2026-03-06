"""Minimal Streamlit app to browse GRPO training episodes.

Usage:
    streamlit run rl/episode_viewer.py
"""

import json
import re
from pathlib import Path

import streamlit as st

RUNS_DIR = Path("runs")


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


def render_scenario(summary: dict | None, turns: list[dict]) -> None:
    if summary:
        sc = summary.get("scenario", {})
        items = sc.get("items", {})
        agent_vals = sc.get("agent_values", {})
        partner_vals = sc.get("partner_values", {})

        st.subheader("Scenario")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Items available**")
            for item, qty in items.items():
                st.markdown(f"- {item.title()}: {qty}")
        with col2:
            st.markdown("**Learner point values**")
            for item, val in agent_vals.items():
                st.markdown(f"- {item.title()}: {val} pts")
            max_learner = sum(items[k] * agent_vals[k] for k in items)
            st.caption(f"Max possible: {max_learner} pts")
        with col3:
            st.markdown("**Partner point values**")
            for item, val in partner_vals.items():
                st.markdown(f"- {item.title()}: {val} pts")
            max_partner = sum(items[k] * partner_vals[k] for k in items)
            st.caption(f"Max possible: {max_partner} pts")

    learner_prompt = next(
        (extract_system_prompt(t["prompt"]) for t in turns if t["agent"] == "learner"),
        None,
    )
    clone_prompt = next(
        (extract_system_prompt(t["prompt"]) for t in turns if t["agent"] == "clone"),
        None,
    )

    if learner_prompt or clone_prompt:
        st.subheader("System Prompts")
        col1, col2 = st.columns(2)
        with col1:
            if learner_prompt:
                st.markdown("**Learner**")
                st.code(learner_prompt, language=None)
        with col2:
            if clone_prompt:
                st.markdown("**Clone**")
                st.code(clone_prompt, language=None)


def render_turn(turn: dict) -> None:
    agent = turn["agent"]
    is_learner = agent == "learner"
    label = "Learner" if is_learner else "Clone"
    icon = "🟦" if is_learner else "🟧"

    response = turn.get("best_response") or turn.get("response", "")
    thought = extract_xml(response, "thought")
    talk = extract_xml(response, "talk")
    action = extract_xml(response, "action")

    st.markdown(f"#### {icon} Turn {turn['turn']} — {label}")

    if talk:
        st.chat_message("user" if is_learner else "assistant").markdown(talk)

    col1, col2 = st.columns(2)
    with col1:
        if thought:
            with st.expander("Thought"):
                st.markdown(thought)
    with col2:
        if action:
            with st.expander("Action"):
                st.code(action, language="json")

    if is_learner and "scores" in turn:
        with st.expander("Candidates & scores"):
            candidates = turn["candidates"]
            scores = turn["scores"]
            advantages = turn.get("advantages", [])
            best = turn.get("best_idx", -1)
            for i, (c, s) in enumerate(zip(candidates, scores)):
                marker = " **← best**" if i == best else ""
                c_talk = extract_xml(c, "talk") or c[:80]
                adv = f"  adv={advantages[i]:+.3f}" if i < len(advantages) else ""
                st.markdown(f"`[{i}]` score={s:.3f}{adv}{marker}  \n> {c_talk}")

    st.divider()


# --- App ---

st.set_page_config(page_title="Episode Viewer", layout="wide")
st.title("GRPO Episode Viewer")

run_dirs = sorted(
    [d for d in RUNS_DIR.iterdir() if d.is_dir() and (d / "episodes").exists()],
    reverse=True,
)

if not run_dirs:
    st.warning("No runs found in `runs/`.")
    st.stop()

run_name = st.sidebar.selectbox("Run", [d.name for d in run_dirs])
run_path = RUNS_DIR / run_name / "episodes"

ep_files = sorted(run_path.glob("ep_*.jsonl"))
if not ep_files:
    st.warning("No episode files found.")
    st.stop()

ep_labels = [f.stem for f in ep_files]
ep_choice = st.sidebar.selectbox("Episode", ep_labels, index=len(ep_labels) - 1)
ep_path = run_path / f"{ep_choice}.jsonl"

turns, summary = load_episode(ep_path)

if summary:
    cols = st.columns(5)
    cols[0].metric("Reward", f"{summary['reward']:.3f}")
    cols[1].metric("Deal", "Yes" if summary["deal"] else "No")
    cols[2].metric("Turns", summary["turns"])
    cols[3].metric("Persona", summary["persona"])
    cols[4].metric("Time", f"{summary['elapsed_s']:.1f}s")

render_scenario(summary, turns)

st.divider()
st.subheader("Dialogue")

for turn in turns:
    render_turn(turn)
