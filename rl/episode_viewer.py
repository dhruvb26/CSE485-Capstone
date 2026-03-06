import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd
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

_NEGOTIATION_ITEMS = ("food", "water", "firewood")


def _thought_char_len(text: str) -> int:
    return len(extract_xml(text, "thought"))


def _action_entropy(candidates: list[str]) -> float:
    """Shannon entropy (bits) over unique <action> strings across candidates."""
    actions = [extract_xml(c, "action") for c in candidates]
    n = len(actions)
    if n <= 1:
        return 0.0
    counts = Counter(actions)
    ent = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def _partner_inference_hit(thought: str, partner_values: dict) -> float:
    """1.0 if the thought correctly names the partner's top-priority item."""
    if not partner_values:
        return 0.0
    true_top = max(_NEGOTIATION_ITEMS, key=lambda i: partner_values.get(i, 0))
    lower = thought.lower()
    for item in _NEGOTIATION_ITEMS:
        patterns = [
            rf"partner.*(?:highest|most|top).*{item}",
            rf"{item}.*partner.*(?:highest|most|top)",
            rf"partner.*priority.*{item}",
        ]
        for pat in patterns:
            if re.search(pat, lower):
                return 1.0 if item == true_top else 0.0
    return 0.0


@st.cache_data(show_spinner="Computing run metrics...")
def compute_run_metrics(run_episodes_dir: str) -> pd.DataFrame:
    """Load every episode in a run and compute the three monitoring signals."""
    ep_dir = Path(run_episodes_dir)
    ep_files = sorted(ep_dir.glob("ep_*.jsonl"))

    rows: list[dict] = []
    for ep_file in ep_files:
        turns, summary = load_episode(ep_file)
        ep_num = int(ep_file.stem.split("_")[1])

        partner_vals: dict = {}
        reward = 0.0
        deal = False
        if summary:
            sc = summary.get("scenario", {})
            partner_vals = sc.get("partner_values", {})
            reward = summary.get("reward", 0.0)
            deal = bool(summary.get("deal"))

        learner_turns = [t for t in turns if t.get("agent") == "learner"]

        thought_lens: list[int] = []
        for t in learner_turns:
            for c in t.get("candidates", []):
                thought_lens.append(_thought_char_len(c))
        avg_thought_len = sum(thought_lens) / len(thought_lens) if thought_lens else 0

        partner_hits: list[float] = []
        for t in learner_turns:
            resp = t.get("best_response", "")
            thought = extract_xml(resp, "thought")
            if thought and partner_vals:
                partner_hits.append(_partner_inference_hit(thought, partner_vals))
        avg_partner_acc = (
            sum(partner_hits) / len(partner_hits) if partner_hits else 0
        )

        entropies: list[float] = []
        for t in learner_turns:
            cands = t.get("candidates", [])
            if len(cands) > 1:
                entropies.append(_action_entropy(cands))
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0

        rows.append(
            {
                "episode": ep_num,
                "thought_length": avg_thought_len,
                "partner_accuracy": avg_partner_acc,
                "action_entropy": avg_entropy,
                "reward": reward,
                "deal": deal,
            }
        )

    return pd.DataFrame(rows)


def render_training_metrics(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        st.info("No episodes found for this run.")
        return

    st.markdown("### :material/monitoring: Thought Trace Length")
    st.caption(
        "Average character count of <thought> across all candidates per episode. "
        "A monotonic decline signals the Echo Trap — reasoning is collapsing."
    )
    st.line_chart(metrics, x="episode", y="thought_length", color="#5B8DEF")

    st.markdown("### :material/psychology: Partner Inference Accuracy")
    st.caption(
        "Fraction of learner turns where the thought correctly identifies "
        "the partner's top-priority item. Flat or declining while reward rises "
        "= reward-blind strategy improvement."
    )
    st.line_chart(metrics, x="episode", y="partner_accuracy", color="#E87461")

    st.markdown("### :material/shuffle: Action Entropy")
    st.caption(
        "Shannon entropy (bits) across the 8 candidate actions per turn. "
        "Near-zero = all candidates collapsed to the same action (self-play collapse)."
    )
    st.line_chart(metrics, x="episode", y="action_entropy", color="#6BC5A0")

    st.divider()
    st.markdown("### :material/finance: Reward & Deal Rate")
    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(metrics, x="episode", y="reward", color="#5B8DEF")
    with col2:
        rolling = metrics.set_index("episode")["deal"].rolling(
            min(20, len(metrics)), min_periods=1
        ).mean().reset_index()
        rolling.columns = ["episode", "deal_rate"]
        st.line_chart(rolling, x="episode", y="deal_rate", color="#E87461")


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

tab_detail, tab_metrics = st.tabs(
    [":material/forum: Episode Detail", ":material/monitoring: Training Metrics"]
)

with tab_detail:
    if summary:
        render_summary(summary)
    render_scenario(summary, turns)
    st.divider()
    st.markdown("### :material/forum: Dialogue")
    for turn in turns:
        render_turn(turn)

with tab_metrics:
    metrics_df = compute_run_metrics(str(run_path))
    render_training_metrics(metrics_df)
