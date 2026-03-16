import argparse
import csv
import json
import random
from pathlib import Path


def load_ji_data(filepath: Path) -> list[dict]:
    """Load JI negotiations from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def filter_completed_negotiations(negotiations: list[dict]) -> list[dict]:
    """Filter to only completed negotiations that have an accepted solution."""
    filtered = []
    for neg in negotiations:
        if neg.get("status") != "completed":
            continue

        solutions = neg.get("solutions", [])
        has_accepted = any(sol.get("accepted", False) for sol in solutions)

        if has_accepted and len(neg.get("comments", [])) > 0:
            filtered.append(neg)

    return filtered


def infer_dialogue_act(text: str) -> str:
    """Infer a dialogue act label from utterance text using heuristics."""
    text_lower = text.lower().strip()

    if any(w in text_lower for w in ["hello", "hi", "hey", "good morning", "good afternoon"]):
        if len(text_lower.split()) <= 5:
            return "<greet>"

    if any(p in text_lower for p in ["deal", "agree", "sounds good", "ok", "okay", "yes", "sure", "perfect", "great"]):
        if any(w in text_lower for w in ["not", "don't", "can't", "won't", "no"]):
            pass
        elif len(text_lower.split()) <= 10:
            return "<agree>"

    if any(p in text_lower for p in ["no deal", "disagree", "can't accept", "won't work", "not acceptable"]):
        return "<disagree>"

    if "?" in text or any(w in text_lower for w in ["what", "how", "why", "which", "where", "when", "could you", "would you", "can you"]):
        return "<inquire>"

    if any(k in text_lower for k in ["salary", "position", "workplace", "company", "holiday", "offer", "propose", "how about", "i'd like", "i want", "i need"]):
        return "<propose>"

    if any(p in text_lower for p in ["important", "priority", "prefer", "value", "need", "want", "my", "i think"]):
        return "<inform>"

    return "<unknown>"


def build_dialogue_acts_row(negotiation: dict) -> dict:
    """Build a dialogue acts row from a negotiation record.

    Concatenates all utterances with ``<sep>`` delimiters and infers
    dialogue act labels for each. Completed negotiations are flagged 0,
    breakdowns are flagged 1.

    Args:
        negotiation: A single negotiation dict containing ``"comments"``
            (list of utterance dicts with ``"body"`` and ``"created_at"``)
            and ``"status"``.

    Returns:
        A dict with ``"text"`` (delimited utterances), ``"meta_text"``
        (string repr of dialogue act list), and ``"flag"`` (0 or 1).
    """
    comments = negotiation.get("comments", [])
    comments = sorted(comments, key=lambda x: x.get("created_at", ""))

    text_parts = []
    meta_parts = []

    for comment in comments:
        body = comment.get("body", "").strip()
        if not body:
            continue

        text_parts.append("<sep>")
        meta_parts.append("<sep>")

        text_parts.append(body.lower())
        meta_parts.append(infer_dialogue_act(body))

    text_parts.append("<end>")
    meta_parts.append("<end>")

    flag = 0 if negotiation.get("status") == "completed" else 1

    return {
        "text": " ".join(text_parts),
        "meta_text": str(meta_parts),
        "flag": flag,
    }


def split_data(
    data: list, train_ratio: float = 0.88, seed: int = 42
) -> tuple[list, list]:
    """Split data into train and test sets."""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def write_json(data: list, filepath: Path):
    """Write data to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_dacts_csv(rows: list[dict], filepath: Path):
    """Write dialogue acts to CSV file."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "meta_text", "flag"])
        writer.writeheader()
        writer.writerows(rows)


def process_ji_dataset(data_dir: Path):
    """Load, filter, split, and write the JI dataset to JSON and CSV formats.

    Filters to completed negotiations with accepted solutions, splits 88/12
    into train/test, and writes both full negotiation JSON files and dialogue
    acts CSVs with heuristic intent labels.

    Args:
        data_dir: Path to the ``ji/`` directory containing ``data.json``.

    Raises:
        FileNotFoundError: If ``data.json`` is missing.
    """
    print("Loading JI data...")
    raw_data = load_ji_data(data_dir / "data.json")
    print(f"Loaded {len(raw_data)} total negotiations")

    print("Filtering to completed negotiations...")
    filtered_data = filter_completed_negotiations(raw_data)
    print(f"Found {len(filtered_data)} completed negotiations with accepted solutions")

    print("Splitting into train/test (88/12)...")
    train_data, test_data = split_data(filtered_data, train_ratio=0.88)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    train_json_path = data_dir / "ji.train.json"
    test_json_path = data_dir / "ji.test.json"

    write_json(train_data, train_json_path)
    print(f"Created {train_json_path}")

    write_json(test_data, test_json_path)
    print(f"Created {test_json_path}")

    print("Building dialogue acts...")
    train_dacts = [build_dialogue_acts_row(neg) for neg in train_data]
    test_dacts = [build_dialogue_acts_row(neg) for neg in test_data]

    train_dacts_path = data_dir / "ji_dacts.train.csv"
    write_dacts_csv(train_dacts, train_dacts_path)
    print(f"Created {train_dacts_path}")

    test_dacts_path = data_dir / "ji_dacts.test.csv"
    write_dacts_csv(test_dacts, test_dacts_path)
    print(f"Created {test_dacts_path}")

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Process JI dataset")
    parser.parse_args()

    data_dir = Path(__file__).parent.parent / "ji"
    process_ji_dataset(data_dir)


if __name__ == "__main__":
    main()
