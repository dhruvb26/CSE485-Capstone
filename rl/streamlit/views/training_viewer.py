import os

import pandas as pd
import streamlit as st

from rl.streamlit.data_loader import load_train_metrics

METRICS_FILENAME = "train_metrics.jsonl"
SFT_DEFAULT_DIR = "checkpoints/sft-tuned"
GRPO_DEFAULT_DIR = "checkpoints/grpo-smoke"

TRAIN_COLOR = "#1f77b4"
VAL_COLOR = "#ff7f0e"

REWARD_COLORS = {
    "format_reward": "#2ca02c",
    "offer_reward": "#d62728",
    "terminal_reward": "#9467bd",
}


def _is_grpo(df: pd.DataFrame) -> bool:
    return "reward" in df.columns and "kl" in df.columns


def _render_sft(df: pd.DataFrame):
    summary_mask = (
        df["train_runtime"].notna()
        if "train_runtime" in df.columns
        else pd.Series(False, index=df.index)
    )

    train_df = df[df["loss"].notna() & ~summary_mask][
        ["global_step", "loss"]
    ].rename(columns={"loss": "train_loss"})
    has_eval = "eval_loss" in df.columns
    eval_df = (
        df[df["eval_loss"].notna() & ~summary_mask][
            ["global_step", "eval_loss"]
        ].rename(columns={"eval_loss": "val_loss"})
        if has_eval
        else pd.DataFrame(columns=["global_step", "val_loss"])
    )

    st.sidebar.markdown(f"**{len(train_df)}** train log entries")
    if has_eval:
        st.sidebar.markdown(f"**{len(eval_df)}** val log entries")

    chart_df = pd.merge(
        train_df, eval_df, on="global_step", how="outer"
    ).sort_values("global_step")

    st.markdown("#### Loss")
    y_cols = ["train_loss"]
    colors = [TRAIN_COLOR]
    if has_eval and not eval_df.empty:
        y_cols.append("val_loss")
        colors.append(VAL_COLOR)
    st.line_chart(chart_df, x="global_step", y=y_cols, color=colors)


def _render_grpo(df: pd.DataFrame):
    summary_mask = (
        df["train_runtime"].notna()
        if "train_runtime" in df.columns
        else pd.Series(False, index=df.index)
    )
    df = df[~summary_mask].copy()

    if df.empty:
        st.info("No GRPO step metrics found.")
        return

    st.sidebar.markdown(f"**{len(df)}** training steps")

    total_steps = int(df["global_step"].max())
    active_steps = int((df["reward_std"] > 0).sum()) if "reward_std" in df.columns else "?"
    mean_reward = df["reward"].mean()
    mean_kl = df["kl"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Steps", total_steps)
    c2.metric("Active steps", f"{active_steps}/{total_steps}")
    c3.metric("Mean reward", f"{mean_reward:.3f}")
    c4.metric("Mean KL", f"{mean_kl:.5f}")

    # --- Reward breakdown ---
    st.markdown("#### Reward breakdown")
    reward_cols = {}
    for name, color in REWARD_COLORS.items():
        col = f"rewards/{name}/mean"
        if col in df.columns:
            reward_cols[col] = color
    if reward_cols:
        st.line_chart(
            df.set_index("global_step")[list(reward_cols.keys())],
            color=list(reward_cols.values()),
        )

    # --- Total reward ---
    st.markdown("#### Total reward")
    reward_chart = df[["global_step"]].copy()
    reward_chart["reward"] = df["reward"]
    colors_list = [TRAIN_COLOR]
    y_list = ["reward"]
    if "reward_std" in df.columns:
        reward_chart["reward_std"] = df["reward_std"]
        y_list.append("reward_std")
        colors_list.append(VAL_COLOR)
    st.line_chart(reward_chart, x="global_step", y=y_list, color=colors_list)

    # --- Policy loss ---
    st.markdown("#### Policy loss")
    st.line_chart(df, x="global_step", y="loss", color=[TRAIN_COLOR])

    # --- KL divergence & entropy ---
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### KL divergence")
        st.line_chart(df, x="global_step", y="kl", color=["#17becf"])
    with col_right:
        st.markdown("#### Entropy")
        st.line_chart(df, x="global_step", y="entropy", color=["#bcbd22"])

    # --- Grad norm & learning rate ---
    col_left2, col_right2 = st.columns(2)
    with col_left2:
        st.markdown("#### Grad norm")
        st.line_chart(df, x="global_step", y="grad_norm", color=["#e377c2"])
    with col_right2:
        st.markdown("#### Learning rate")
        st.line_chart(df, x="global_step", y="learning_rate", color=["#7f7f7f"])

    # --- Completion length ---
    if "completions/mean_length" in df.columns:
        st.markdown("#### Completion length")
        len_df = df[["global_step"]].copy()
        len_df["mean"] = df["completions/mean_length"]
        if "completions/min_length" in df.columns:
            len_df["min"] = df["completions/min_length"]
        if "completions/max_length" in df.columns:
            len_df["max"] = df["completions/max_length"]
        st.line_chart(len_df, x="global_step", y=[c for c in len_df.columns if c != "global_step"])

    # --- Dead steps ---
    if "frac_reward_zero_std" in df.columns:
        st.markdown("#### Fraction of zero-std reward steps")
        st.line_chart(df, x="global_step", y="frac_reward_zero_std", color=["#d62728"])


def render():
    """Main entry point for the Training Metrics view."""
    st.sidebar.markdown("## Configuration")

    mode = st.sidebar.radio("Training type", ["SFT", "GRPO"], horizontal=True)

    default_dir = SFT_DEFAULT_DIR if mode == "SFT" else GRPO_DEFAULT_DIR
    output_dir = st.sidebar.text_input("Checkpoint directory", value=default_dir)
    metrics_path = os.path.join(output_dir, METRICS_FILENAME)

    df = load_train_metrics(metrics_path)

    if df is None or df.empty:
        st.info(f"No training metrics found at `{metrics_path}`.")
        return

    if mode == "GRPO" or _is_grpo(df):
        _render_grpo(df)
    else:
        _render_sft(df)
