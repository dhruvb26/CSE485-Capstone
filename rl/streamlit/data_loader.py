"""Cached data-loading helpers for the Streamlit app."""

import json
import os

import pandas as pd
import streamlit as st

from rl.sft.data import count_conversations, load_conversation


@st.cache_data
def count_rows(csv_path: str) -> int:
    return count_conversations(csv_path)


@st.cache_data
def load_single(csv_path: str, row_idx: int):
    return load_conversation(csv_path, row_idx)


@st.cache_data
def load_annotations(jsonl_path: str) -> dict[str, list[dict]]:
    if not os.path.exists(jsonl_path):
        return {}
    records: dict[str, list[dict]] = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            key = rec.get(
                "id",
                f"{rec.get('dataset', 'unknown')}:{rec['row_idx']}:{rec['agent_id']}",
            )
            records[key] = rec.get("messages", rec.get("prompt", []))
    return records


@st.cache_data
def load_train_metrics(jsonl_path: str) -> pd.DataFrame | None:
    """Load training metrics JSONL into a DataFrame, or None if missing."""
    if not os.path.exists(jsonl_path):
        return None
    rows: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            rows.append(json.loads(line))
    if not rows:
        return None
    return pd.DataFrame(rows)
