import os

import pandas as pd
import streamlit as st

from rl.streamlit.data_loader import load_train_metrics

METRICS_FILENAME = "train_metrics.jsonl"
DEFAULT_OUTPUT_DIR = "checkpoints/sft-tuned"

TRAIN_COLOR = "#1f77b4"
VAL_COLOR = "#ff7f0e"


def render():
    """Main entry point for the Training Metrics view."""
    st.sidebar.markdown("## Configuration")
    output_dir = st.sidebar.text_input("Checkpoint directory", value=DEFAULT_OUTPUT_DIR)
    metrics_path = os.path.join(output_dir, METRICS_FILENAME)

    df = load_train_metrics(metrics_path)

    if df is None or df.empty:
        st.info("No training metrics found yet.")
        return

    summary_mask = df["train_runtime"].notna() if "train_runtime" in df.columns else pd.Series(False, index=df.index)

    train_df = df[df["loss"].notna() & ~summary_mask][["global_step", "loss"]].rename(columns={"loss": "train_loss"})
    has_eval = "eval_loss" in df.columns
    eval_df = (
        df[df["eval_loss"].notna() & ~summary_mask][["global_step", "eval_loss"]].rename(columns={"eval_loss": "val_loss"})
        if has_eval
        else pd.DataFrame(columns=["global_step", "val_loss"])
    )

    st.sidebar.markdown(f"**{len(train_df)}** train log entries")
    if has_eval:
        st.sidebar.markdown(f"**{len(eval_df)}** val log entries")

    chart_df = pd.merge(train_df, eval_df, on="global_step", how="outer").sort_values("global_step")

    st.markdown("#### Loss")
    y_cols = ["train_loss"]
    colors = [TRAIN_COLOR]
    if has_eval and not eval_df.empty:
        y_cols.append("val_loss")
        colors.append(VAL_COLOR)
    st.line_chart(chart_df, x="global_step", y=y_cols, color=colors)
