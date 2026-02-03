"""
Script to convert JobInterview (JI) dataset files into structured formats.

Input files:
- ji/data.json: Raw negotiation data with users, comments, solutions

Output files:
1. Main JI JSON format:
   - ji/ji.train.json: Training negotiations (88%)
   - ji/ji.test.json: Test negotiations (12%)

2. Dialogue Acts CSV format:
   - ji/ji_dacts.train.csv: Training dialogue acts
   - ji/ji_dacts.test.csv: Test dialogue acts

Usage:
    python ji.py              # Process dataset and create all files
    python ji.py --example    # Extract first entry to example JSON files
"""

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
    """
    Filter to only completed negotiations with an accepted solution.
    """
    filtered = []
    for neg in negotiations:
        if neg.get("status") != "completed":
            continue

        # Check for accepted solution
        solutions = neg.get("solutions", [])
        has_accepted = any(sol.get("accepted", False) for sol in solutions)

        if has_accepted and len(neg.get("comments", [])) > 0:
            filtered.append(neg)

    return filtered


def infer_dialogue_act(text: str) -> str:
    """
    Infer dialogue act from utterance text.
    This is a heuristic - actual annotations would be more accurate.
    """
    text_lower = text.lower().strip()

    # Greetings
    if any(
        word in text_lower
        for word in ["hello", "hi", "hey", "good morning", "good afternoon"]
    ):
        if len(text_lower.split()) <= 5:
            return "<greet>"

    # Agreement
    if any(
        phrase in text_lower
        for phrase in [
            "deal",
            "agree",
            "sounds good",
            "ok",
            "okay",
            "yes",
            "sure",
            "perfect",
            "great",
        ]
    ):
        if any(word in text_lower for word in ["not", "don't", "can't", "won't", "no"]):
            pass  # Negation present, not agreement
        elif len(text_lower.split()) <= 10:
            return "<agree>"

    # Disagreement
    if any(
        phrase in text_lower
        for phrase in [
            "no deal",
            "disagree",
            "can't accept",
            "won't work",
            "not acceptable",
        ]
    ):
        return "<disagree>"

    # Questions/Inquiries
    if "?" in text or any(
        word in text_lower
        for word in [
            "what",
            "how",
            "why",
            "which",
            "where",
            "when",
            "could you",
            "would you",
            "can you",
        ]
    ):
        return "<inquire>"

    # Proposals (mentions of specific values/options)
    proposal_keywords = [
        "salary",
        "position",
        "workplace",
        "company",
        "holiday",
        "offer",
        "propose",
        "how about",
        "i'd like",
        "i want",
        "i need",
    ]
    if any(keyword in text_lower for keyword in proposal_keywords):
        return "<propose>"

    # Information sharing
    if any(
        phrase in text_lower
        for phrase in [
            "important",
            "priority",
            "prefer",
            "value",
            "need",
            "want",
            "my",
            "i think",
        ]
    ):
        return "<inform>"

    return "<unknown>"


def build_dialogue_acts_row(negotiation: dict) -> dict:
    """
    Build a dialogue acts row for a negotiation.

    Returns:
        {
            'text': '<sep> utterance1 <sep> utterance2 ... <end>',
            'meta_text': "['<sep>', '<act1>', '<sep>', '<act2>', ..., '<end>']",
            'flag': 0  # No breakdown for completed negotiations
        }
    """
    comments = negotiation.get("comments", [])

    # Sort comments by created_at
    comments = sorted(comments, key=lambda x: x.get("created_at", ""))

    text_parts = []
    meta_parts = []

    for comment in comments:
        body = comment.get("body", "").strip()
        if not body:
            continue

        # Add separator
        text_parts.append("<sep>")
        meta_parts.append("<sep>")

        # Add utterance and its dialogue act
        text_parts.append(body.lower())
        act = infer_dialogue_act(body)
        meta_parts.append(act)

    # Add end marker
    text_parts.append("<end>")
    meta_parts.append("<end>")

    # Build text string
    text = " ".join(text_parts)

    # Build meta_text as string representation of list
    meta_text = str(meta_parts)

    # Flag: 0 for completed, 1 for breakdown
    flag = 0 if negotiation.get("status") == "completed" else 1

    return {"text": text, "meta_text": meta_text, "flag": flag}


def split_data(
    data: list, train_ratio: float = 0.88, seed: int = 42
) -> tuple[list, list]:
    """Split data into train and test sets."""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_data = shuffled[:split_idx]
    test_data = shuffled[split_idx:]

    return train_data, test_data


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
    """Process JI dataset and create all output files."""

    # Load data
    print("Loading JI data...")
    raw_data = load_ji_data(data_dir / "data.json")
    print(f"Loaded {len(raw_data)} total negotiations")

    # Filter to completed negotiations with accepted solutions
    print("Filtering to completed negotiations...")
    filtered_data = filter_completed_negotiations(raw_data)
    print(f"Found {len(filtered_data)} completed negotiations with accepted solutions")

    # Split into train/test (88/12)
    print("Splitting into train/test (88/12)...")
    train_data, test_data = split_data(filtered_data, train_ratio=0.88)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Write main JSON files
    train_json_path = data_dir / "ji.train.json"
    test_json_path = data_dir / "ji.test.json"

    write_json(train_data, train_json_path)
    print(f"Created {train_json_path}")

    write_json(test_data, test_json_path)
    print(f"Created {test_json_path}")

    # Build and write dialogue acts CSVs
    print("Building dialogue acts...")
    train_dacts = [build_dialogue_acts_row(neg) for neg in train_data]
    test_dacts = [build_dialogue_acts_row(neg) for neg in test_data]

    train_dacts_path = data_dir / "ji_dacts.train.csv"
    test_dacts_path = data_dir / "ji_dacts.test.csv"

    write_dacts_csv(train_dacts, train_dacts_path)
    print(f"Created {train_dacts_path}")

    write_dacts_csv(test_dacts, test_dacts_path)
    print(f"Created {test_dacts_path}")

    print("\nDone!")


def extract_examples(data_dir: Path, examples_dir: Path):
    """Extract first entry to example JSON files."""
    examples_dir.mkdir(parents=True, exist_ok=True)

    # Main JSON example
    train_json = data_dir / "ji.train.json"
    if train_json.exists():
        with open(train_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data:
            example = data[0]
            with open(
                examples_dir / "negotiation_example.json", "w", encoding="utf-8"
            ) as f:
                json.dump(example, f, indent=2, ensure_ascii=False)
            print(f"Saved {examples_dir / 'negotiation_example.json'}")

    # Dialogue acts example
    train_dacts = data_dir / "ji_dacts.train.csv"
    if train_dacts.exists():
        with open(train_dacts, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        example = {
            "text": row["text"],
            "meta_text": eval(row["meta_text"]),  # Convert string repr to list
            "flag": int(row["flag"]),
        }

        with open(examples_dir / "dacts_example.json", "w", encoding="utf-8") as f:
            json.dump(example, f, indent=2, ensure_ascii=False)
        print(f"Saved {examples_dir / 'dacts_example.json'}")

    print(f"\nExtracted examples to {examples_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Process JobInterview dataset or extract example data"
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Extract first entry to example JSON files",
    )

    args = parser.parse_args()

    # Set paths relative to this script
    script_dir = Path(__file__).parent
    data_dir = script_dir / "ji"
    examples_dir = data_dir / "examples"

    if args.example:
        train_json = data_dir / "ji.train.json"
        if not train_json.exists():
            print("JSON files not found. Run without --example first to generate them.")
            return
        extract_examples(data_dir, examples_dir)
    else:
        process_ji_dataset(data_dir)


if __name__ == "__main__":
    main()
