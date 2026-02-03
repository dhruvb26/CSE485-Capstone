"""
Script to convert Casino dataset files into a structured CSV format.

Input files:
- casino/utterances.jsonl: Individual utterances with text, speaker, and annotations
- casino/conversations.json: Conversation metadata with participant preferences and outcomes
- casino/speakers.json: Speaker demographics and personality info

Output:
- casino/ca.train.csv: Training data (88%)
- casino/ca.test.csv: Test data (12%)

CSV columns:
    - chat_logs: JSON list of messages with text, task_data, and speaker id
    - participant_info: JSON dict with participant preferences, outcomes, demographics, personality
    - annotations: JSON list of [text, annotation] pairs

Usage:
    python casino.py              # Process dataset and create train/test CSVs
    python casino.py --example    # Extract first row to example JSON files
"""

import argparse
import json
import csv
import random
from pathlib import Path
from collections import defaultdict


def load_utterances(filepath: Path) -> dict[int, list[dict]]:
    """Load utterances and group by dialogue_id."""
    dialogues = defaultdict(list)

    with open(filepath, "r") as f:
        for line in f:
            utterance = json.loads(line.strip())
            dialogue_id = utterance["meta"]["dialogue_id"]
            dialogues[dialogue_id].append(utterance)

    # Sort utterances within each dialogue by their id (utterance_X)
    for dialogue_id in dialogues:
        dialogues[dialogue_id].sort(key=lambda x: int(x["id"].split("_")[1]))

    return dict(dialogues)


def load_conversations(filepath: Path) -> dict[int, dict]:
    """Load conversations and map by dialogue_id."""
    with open(filepath, "r") as f:
        data = json.load(f)

    # Map by dialogue_id
    conversations = {}
    for conv_id, conv_data in data.items():
        dialogue_id = conv_data["meta"]["dialogue_id"]
        conversations[dialogue_id] = conv_data["meta"]

    return conversations


def load_speakers(filepath: Path) -> dict[str, dict]:
    """Load speaker demographics and personality."""
    with open(filepath, "r") as f:
        data = json.load(f)

    speakers = {}
    for speaker_id, speaker_data in data.items():
        speakers[speaker_id] = speaker_data["meta"]

    return speakers


def build_chat_logs(utterances: list[dict]) -> list[dict]:
    """
    Build chat_logs format:
    [
        {
            "text": "...",
            "task_data": {
                "data": "",
                "issue2youget": {"Firewood": "", "Water": "", "Food": ""},
                "issue2theyget": {"Firewood": "", "Water": "", "Food": ""}
            },
            "id": "mturk_agent_1"
        },
        ...
    ]
    """
    chat_logs = []

    for utt in utterances:
        meta = utt["meta"]
        speaker_internal_id = meta["speaker_internal_id"]

        # Build task_data
        task_data = {
            "data": "",
            "issue2youget": {"Firewood": "", "Water": "", "Food": ""},
            "issue2theyget": {"Firewood": "", "Water": "", "Food": ""},
        }

        # Check if this is a Submit-Deal, Accept-Deal, or Reject-Deal
        if "issue2youget" in meta:
            task_data["issue2youget"] = meta["issue2youget"]
        if "issue2theyget" in meta:
            task_data["issue2theyget"] = meta["issue2theyget"]
        if "data" in meta:
            task_data["data"] = meta["data"]

        chat_entry = {
            "text": utt["text"],
            "task_data": task_data,
            "id": speaker_internal_id,
        }

        chat_logs.append(chat_entry)

    return chat_logs


def build_participant_info(
    conversation_meta: dict, utterances: list[dict], speakers: dict[str, dict]
) -> dict:
    """
    Build participant_info format:
    {
        "mturk_agent_1": {
            "value2issue": {...},
            "value2reason": {...},
            "outcomes": {...},
            "demographics": {...},
            "personality": {...}
        },
        "mturk_agent_2": {...}
    }
    """
    participant_info = {}

    # Get base participant info from conversation
    conv_participant_info = conversation_meta.get("participant_info", {})

    # Build a mapping from speaker_internal_id to speaker_id
    internal_to_speaker = {}
    for utt in utterances:
        internal_id = utt["meta"]["speaker_internal_id"]
        speaker_id = utt["meta"]["speaker_id"]
        internal_to_speaker[internal_id] = speaker_id

    for agent_id in ["mturk_agent_1", "mturk_agent_2"]:
        agent_info = conv_participant_info.get(agent_id, {})

        # Get speaker demographics and personality
        speaker_id = internal_to_speaker.get(agent_id)
        speaker_data = speakers.get(speaker_id, {})

        participant_info[agent_id] = {
            "value2issue": agent_info.get("value2issue", {}),
            "value2reason": agent_info.get("value2reason", {}),
            "outcomes": agent_info.get("outcomes", {}),
            "demographics": speaker_data.get("demographics", {}),
            "personality": speaker_data.get("personality", {}),
        }

    return participant_info


def build_annotations(utterances: list[dict]) -> list[list[str]]:
    """
    Build annotations format:
    [
        ["Hello...", "self-need,other-need"],
        ["Submit-Deal", "non-strategic"],
        ...
    ]
    """
    annotations = []

    for utt in utterances:
        text = utt["text"]
        annotation = utt["meta"].get("annotations")

        # Handle None annotations (for Submit-Deal, Accept-Deal, Reject-Deal)
        if annotation is None:
            annotation = "non-strategic"

        annotations.append([text, annotation])

    return annotations


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


def write_csv(rows: list[dict], filepath: Path):
    """Write rows to CSV file."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["chat_logs", "participant_info", "annotations"]
        )
        writer.writeheader()
        writer.writerows(rows)


def process_casino_dataset(data_dir: Path):
    """Process the casino dataset and create train/test CSV files."""

    # Load all data
    print("Loading utterances...")
    utterances_by_dialogue = load_utterances(data_dir / "utterances.jsonl")

    print("Loading conversations...")
    conversations = load_conversations(data_dir / "conversations.json")

    print("Loading speakers...")
    speakers = load_speakers(data_dir / "speakers.json")

    print(f"Found {len(utterances_by_dialogue)} dialogues")

    # Process each dialogue
    rows = []
    for dialogue_id in sorted(utterances_by_dialogue.keys()):
        utterances = utterances_by_dialogue[dialogue_id]
        conversation_meta = conversations.get(dialogue_id, {"participant_info": {}})

        # Build the three columns
        chat_logs = build_chat_logs(utterances)
        participant_info = build_participant_info(
            conversation_meta, utterances, speakers
        )
        annotations = build_annotations(utterances)

        rows.append(
            {
                "chat_logs": json.dumps(chat_logs),
                "participant_info": json.dumps(participant_info),
                "annotations": json.dumps(annotations),
            }
        )

    # Split into train/test (88/12)
    print("Splitting into train/test (88/12)...")
    train_rows, test_rows = split_data(rows, train_ratio=0.88)
    print(f"Train: {len(train_rows)}, Test: {len(test_rows)}")

    # Write train CSV
    train_path = data_dir / "ca.train.csv"
    write_csv(train_rows, train_path)
    print(f"Created {train_path}")

    # Write test CSV
    test_path = data_dir / "ca.test.csv"
    write_csv(test_rows, test_path)
    print(f"Created {test_path}")

    print("Done!")


def extract_example(csv_path: Path, examples_dir: Path, row_index: int = 0):
    """
    Extract a row from the CSV and save each column as a separate JSON file.

    Args:
        csv_path: Path to the processed CSV file
        examples_dir: Directory to save example files
        row_index: Which row to extract (0-indexed, default is first row)
    """
    # Create examples directory if it doesn't exist
    examples_dir.mkdir(parents=True, exist_ok=True)

    # Read the CSV
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if row_index >= len(rows):
        print(f"Error: Row index {row_index} out of range (only {len(rows)} rows)")
        return

    row = rows[row_index]

    # Parse and save each column as formatted JSON
    columns = ["chat_logs", "participant_info", "annotations"]

    for col in columns:
        json_data = json.loads(row[col])
        output_file = examples_dir / f"{col}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"Saved {output_file}")

    print(f"\nExtracted row {row_index} to {examples_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Process Casino dataset or extract example data"
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Extract first row from CSV to example JSON files",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=0,
        help="Row index to extract (0-indexed, default: 0)",
    )

    args = parser.parse_args()

    # Set paths relative to this script
    script_dir = Path(__file__).parent
    data_dir = script_dir / "casino"
    train_path = data_dir / "ca.train.csv"
    examples_dir = data_dir / "examples"

    if args.example:
        if not train_path.exists():
            print(f"CSV file not found at {train_path}")
            print("Run without --example first to generate the CSV")
            return
        extract_example(train_path, examples_dir, args.row)
    else:
        process_casino_dataset(data_dir)


if __name__ == "__main__":
    main()
