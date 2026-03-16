import argparse
import json
import csv
import random
from pathlib import Path
from collections import defaultdict


def load_utterances(filepath: Path) -> dict[int, list[dict]]:
    """Load utterances from JSONL and group by dialogue_id, sorted by turn order."""
    dialogues = defaultdict(list)

    with open(filepath, "r") as f:
        for line in f:
            utterance = json.loads(line.strip())
            dialogue_id = utterance["meta"]["dialogue_id"]
            dialogues[dialogue_id].append(utterance)

    for dialogue_id in dialogues:
        dialogues[dialogue_id].sort(key=lambda x: int(x["id"].split("_")[1]))

    return dict(dialogues)


def load_conversations(filepath: Path) -> dict[int, dict]:
    """Load conversations and map by dialogue_id."""
    with open(filepath, "r") as f:
        data = json.load(f)

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
    """Build structured chat log entries from raw utterances.

    Each entry contains the utterance text, task_data (deal allocations if
    applicable), and the speaker's internal id. Deal-related metadata
    (issue2youget, issue2theyget, data) is preserved when present.

    Args:
        utterances: List of raw utterance dicts from the CaSiNo corpus,
            each containing ``"text"`` and ``"meta"`` with speaker and
            deal information.

    Returns:
        A list of chat log dicts, each with keys ``"text"``,
        ``"task_data"``, and ``"id"``.
    """
    chat_logs = []

    for utt in utterances:
        meta = utt["meta"]
        speaker_internal_id = meta["speaker_internal_id"]

        task_data = {
            "data": "",
            "issue2youget": {"Firewood": "", "Water": "", "Food": ""},
            "issue2theyget": {"Firewood": "", "Water": "", "Food": ""},
        }

        if "issue2youget" in meta:
            task_data["issue2youget"] = meta["issue2youget"]
        if "issue2theyget" in meta:
            task_data["issue2theyget"] = meta["issue2theyget"]
        if "data" in meta:
            task_data["data"] = meta["data"]

        chat_logs.append({
            "text": utt["text"],
            "task_data": task_data,
            "id": speaker_internal_id,
        })

    return chat_logs


def build_participant_info(
    conversation_meta: dict, utterances: list[dict], speakers: dict[str, dict]
) -> dict:
    """Build participant info combining negotiation preferences with speaker metadata.

    Merges each agent's value-to-issue mappings and negotiation outcomes from
    the conversation metadata with demographics and personality traits from
    the speakers data.

    Args:
        conversation_meta: Conversation-level metadata containing
            ``"participant_info"`` with value2issue, value2reason, and outcomes.
        utterances: Raw utterances used to map internal agent ids to speaker ids.
        speakers: Speaker metadata keyed by speaker id, containing demographics
            and personality.

    Returns:
        A dict keyed by agent id (``"mturk_agent_1"``, ``"mturk_agent_2"``),
        each containing ``"value2issue"``, ``"value2reason"``, ``"outcomes"``,
        ``"demographics"``, and ``"personality"``.
    """
    participant_info = {}

    conv_participant_info = conversation_meta.get("participant_info", {})

    internal_to_speaker = {}
    for utt in utterances:
        internal_id = utt["meta"]["speaker_internal_id"]
        speaker_id = utt["meta"]["speaker_id"]
        internal_to_speaker[internal_id] = speaker_id

    for agent_id in ["mturk_agent_1", "mturk_agent_2"]:
        agent_info = conv_participant_info.get(agent_id, {})

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
    """Build [text, annotation] pairs from utterances."""
    annotations = []

    for utt in utterances:
        text = utt["text"]
        annotation = utt["meta"].get("annotations")

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
    return shuffled[:split_idx], shuffled[split_idx:]


def write_csv(rows: list[dict], filepath: Path):
    """Write rows to CSV file."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["chat_logs", "participant_info", "annotations"]
        )
        writer.writeheader()
        writer.writerows(rows)


def process_casino_dataset(data_dir: Path):
    """Load, process, and split the CaSiNo corpus into train/test CSVs.

    Reads utterances, conversations, and speakers from the corpus directory,
    builds structured chat_logs, participant_info, and annotations for each
    dialogue, then writes an 88/12 train/test split.

    Args:
        data_dir: Path to the ``casino/`` directory containing
            ``utterances.jsonl``, ``conversations.json``, and
            ``speakers.json``.

    Raises:
        FileNotFoundError: If any of the required input files are missing.
    """
    print("Loading utterances...")
    utterances_by_dialogue = load_utterances(data_dir / "utterances.jsonl")

    print("Loading conversations...")
    conversations = load_conversations(data_dir / "conversations.json")

    print("Loading speakers...")
    speakers = load_speakers(data_dir / "speakers.json")

    print(f"Found {len(utterances_by_dialogue)} dialogues")

    rows = []
    for dialogue_id in sorted(utterances_by_dialogue.keys()):
        utterances = utterances_by_dialogue[dialogue_id]
        conversation_meta = conversations.get(dialogue_id, {"participant_info": {}})

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

    print("Splitting into train/test (88/12)...")
    train_rows, test_rows = split_data(rows, train_ratio=0.88)
    print(f"Train: {len(train_rows)}, Test: {len(test_rows)}")

    train_path = data_dir / "ca.train.csv"
    write_csv(train_rows, train_path)
    print(f"Created {train_path}")

    test_path = data_dir / "ca.test.csv"
    write_csv(test_rows, test_path)
    print(f"Created {test_path}")

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Process Casino dataset")
    parser.parse_args()

    data_dir = Path(__file__).parent.parent / "casino"
    process_casino_dataset(data_dir)


if __name__ == "__main__":
    main()
