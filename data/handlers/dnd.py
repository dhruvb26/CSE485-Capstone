import argparse
import csv
import json
import re
import uuid
from pathlib import Path


def parse_tagged_line(line: str) -> dict | None:
    """Parse a tagged DND line into its four sections.

    Args:
        line: A single line from train.txt/test.txt with ``<input>``,
            ``<dialogue>``, ``<output>``, and ``<partner_input>`` tags.

    Returns:
        A dict with keys ``"input"``, ``"dialogue"``, ``"output"``, and
        ``"partner_input"``, or None if any tag is missing.
    """
    input_match = re.search(r"<input>\s*(.*?)\s*</input>", line)
    dialogue_match = re.search(r"<dialogue>\s*(.*?)\s*</dialogue>", line)
    output_match = re.search(r"<output>\s*(.*?)\s*</output>", line)
    partner_input_match = re.search(r"<partner_input>\s*(.*?)\s*</partner_input>", line)

    if not all([input_match, dialogue_match, output_match, partner_input_match]):
        return None

    return {
        "input": input_match.group(1).strip(),
        "dialogue": dialogue_match.group(1).strip(),
        "output": output_match.group(1).strip(),
        "partner_input": partner_input_match.group(1).strip(),
    }


def parse_input_values(input_str: str) -> dict:
    """Parse the space-separated input string into counts and values.

    Args:
        input_str: Six space-separated integers in the format
            ``"count_book value_book count_hat value_hat count_ball value_ball"``.

    Returns:
        A dict with ``"count"`` and ``"value"`` lists of length 3
        (books, hats, balls). Empty lists if the format is invalid.
    """
    parts = input_str.split()
    if len(parts) != 6:
        return {"count": [], "value": []}

    count_book, value_book = int(parts[0]), int(parts[1])
    count_hat, value_hat = int(parts[2]), int(parts[3])
    count_ball, value_ball = int(parts[4]), int(parts[5])

    return {
        "count": [count_book, count_hat, count_ball],
        "value": [value_book, value_hat, value_ball],
    }


def parse_output_allocation(output_str: str) -> tuple[dict, dict] | tuple[None, None]:
    """Parse the output allocation string into YOU and THEM item splits.

    Args:
        output_str: Allocation string like
            ``"item0=0 item1=4 item2=0 item0=1 item1=0 item2=1"`` where
            the first 3 are YOU's allocation and the last 3 are THEM's.
            May contain ``<disagree>`` for failed negotiations.

    Returns:
        A tuple of (you_alloc, them_alloc) dicts with keys ``"book"``,
        ``"hat"``, ``"ball"``, or (None, None) if no deal was reached.
    """
    if "<disagree>" in output_str or "no agreement" in output_str.lower():
        return None, None

    items = re.findall(r"item(\d)=(\d+)", output_str)
    if len(items) != 6:
        return None, None

    you_alloc = {
        "book": int(items[0][1]),
        "hat": int(items[1][1]),
        "ball": int(items[2][1]),
    }
    them_alloc = {
        "book": int(items[3][1]),
        "hat": int(items[4][1]),
        "ball": int(items[5][1]),
    }

    return you_alloc, them_alloc


def create_basic_csv(input_path: Path, output_path: Path) -> int:
    """Create a basic DND CSV with input, dialogue, output, partner_input columns.

    Args:
        input_path: Path to a DND text file (e.g. ``train.txt``).
        output_path: Path to write the output CSV.

    Returns:
        The number of rows written.
    """
    rows = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parsed = parse_tagged_line(line)
            if not parsed:
                continue

            input_json = json.dumps(parse_input_values(parsed["input"]))
            partner_input_json = json.dumps(parse_input_values(parsed["partner_input"]))

            rows.append(
                {
                    "input": input_json,
                    "dialogue": parsed["dialogue"],
                    "output": parsed["output"],
                    "partner_input": partner_input_json,
                }
            )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["input", "dialogue", "output", "partner_input"]
        )
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def create_annotated_csv(input_path: Path, output_path: Path) -> int:
    """Create an annotated DND CSV with scenario, outcome, and event structures.

    Parses each negotiation into a structured format with UUIDs, knowledge
    bases for both agents, deal outcomes with reward calculations, and
    dialogue events with heuristic intent labels.

    Args:
        input_path: Path to a DND text file (e.g. ``train.txt``).
        output_path: Path to write the annotated CSV.

    Returns:
        The number of rows written.
    """
    rows = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parsed = parse_tagged_line(line)
            if not parsed:
                continue

            you_input = parse_input_values(parsed["input"])
            them_input = parse_input_values(parsed["partner_input"])
            you_alloc, them_alloc = parse_output_allocation(parsed["output"])

            nego_uuid = f"E_{uuid.uuid4().hex[:16]}"
            scenario_uuid = f"FB_{uuid.uuid4().hex[:16]}"

            scenario = {
                "attributes": [],
                "uuid": scenario_uuid,
                "kbs": [
                    [
                        {"Count": you_input["count"][0], "Name": "book", "Value": you_input["value"][0]},
                        {"Count": you_input["count"][1], "Name": "hat", "Value": you_input["value"][1]},
                        {"Count": you_input["count"][2], "Name": "ball", "Value": you_input["value"][2]},
                    ],
                    [
                        {"Count": them_input["count"][0], "Name": "book", "Value": them_input["value"][0]},
                        {"Count": them_input["count"][1], "Name": "hat", "Value": them_input["value"][1]},
                        {"Count": them_input["count"][2], "Name": "ball", "Value": them_input["value"][2]},
                    ],
                ],
            }

            valid_deal = you_alloc is not None
            if valid_deal:
                you_reward = (
                    you_alloc["book"] * you_input["value"][0]
                    + you_alloc["hat"] * you_input["value"][1]
                    + you_alloc["ball"] * you_input["value"][2]
                )
                them_reward = (
                    them_alloc["book"] * them_input["value"][0]
                    + them_alloc["hat"] * them_input["value"][1]
                    + them_alloc["ball"] * them_input["value"][2]
                )
                outcome = {
                    "valid_deal": True,
                    "item_split": [you_alloc, them_alloc],
                    "reward": {"0": you_reward, "1": them_reward},
                    "agreed": True,
                }
            else:
                outcome = {
                    "valid_deal": False,
                    "item_split": [
                        {"book": 0, "hat": 0, "ball": 0},
                        {"book": 0, "hat": 0, "ball": 0},
                    ],
                    "reward": {"0": 0, "1": 0},
                    "agreed": False,
                }

            events = parse_dialogue_to_events(parsed["dialogue"], you_alloc)

            rows.append(
                {
                    "uuid": nego_uuid,
                    "scenario": json.dumps(scenario),
                    "agents_info": json.dumps({}),
                    "scenario_uuid": scenario_uuid,
                    "agents": json.dumps({"0": "human", "1": "human"}),
                    "outcome": json.dumps(outcome),
                    "events": json.dumps(events),
                }
            )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "uuid", "scenario", "agents_info",
                "scenario_uuid", "agents", "outcome", "events",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def parse_dialogue_to_events(dialogue: str, final_alloc: dict | None) -> list[dict]:
    """Parse a DND dialogue string into a list of structured event dicts.

    Splits the dialogue on ``<eos>`` tokens, assigns speaker ids (0 for YOU,
    1 for THEM), and handles ``<selection>`` events. Intent labels are
    inferred heuristically via :func:`infer_basic_intent`.

    Args:
        dialogue: Raw dialogue string in the format
            ``"YOU: msg <eos> THEM: msg <eos> ... <selection>"``.
        final_alloc: The final item allocation for agent 0 (YOU), or None
            if no deal was reached.

    Returns:
        A list of event dicts, each with ``"agent"``, ``"time"``,
        ``"action"``, ``"data"``, and ``"metadata"`` keys.
    """
    events = []
    turns = dialogue.split("<eos>")

    for i, turn in enumerate(turns):
        turn = turn.strip()
        if not turn:
            continue

        if "<selection>" in turn:
            if turn.startswith("YOU:"):
                agent = 0
                remaining = turn[4:].strip()
            elif turn.startswith("THEM:"):
                agent = 1
                remaining = turn[5:].strip()
            else:
                if "YOU:" in turns[i - 1] if i > 0 else False:
                    agent = 1
                else:
                    agent = 0
                remaining = turn

            text_before = remaining.replace("<selection>", "").strip()
            if text_before:
                events.append({
                    "agent": agent,
                    "time": len(events),
                    "action": "message",
                    "data": text_before,
                    "metadata": {"intent": "unknown", "proposal": None, "proposal_type": None},
                })

            if final_alloc:
                events.append({
                    "agent": agent,
                    "time": len(events),
                    "action": "select",
                    "data": final_alloc if agent == 0 else None,
                    "metadata": {"intent": "select"},
                })
        else:
            if turn.startswith("YOU:"):
                agent = 0
                text = turn[4:].strip()
            elif turn.startswith("THEM:"):
                agent = 1
                text = turn[5:].strip()
            else:
                continue

            intent = infer_basic_intent(text)
            events.append({
                "agent": agent,
                "time": len(events),
                "action": "message",
                "data": text,
                "metadata": {"intent": intent, "proposal": None, "proposal_type": None},
            })

    return events


def infer_basic_intent(text: str) -> str:
    """Infer a basic dialogue act label from utterance text using heuristics."""
    text_lower = text.lower()

    if any(w in text_lower for w in ["deal", "ok", "okay", "yes", "agree", "sounds good", "sure"]):
        if any(w in text_lower for w in ["no deal", "no agreement"]):
            return "disagree"
        return "agree"

    if any(w in text_lower for w in ["no", "can't", "won't", "cannot", "not going to", "not possible"]):
        return "disagree"

    if "?" in text:
        return "inquire"

    if any(w in text_lower for w in ["hi", "hello", "hey"]):
        return "greet"

    if any(w in text_lower for w in ["i'd like", "i want", "i need", "i'll take", "how about", "give me", "can i have"]):
        return "propose"

    return "unknown"


def process_dnd_dataset(data_dir: Path):
    """Process all DND text files into basic and annotated CSV formats.

    Args:
        data_dir: Path to the ``dnd/`` directory containing ``train.txt``
            and ``test.txt``.
    """
    train_input = data_dir / "train.txt"
    if train_input.exists():
        print("Processing train split...")
        basic_path = data_dir / "dnd.train.csv"
        count = create_basic_csv(train_input, basic_path)
        print(f"Created {basic_path} ({count} rows)")

        ann_path = data_dir / "dnd_ann.train.csv"
        count = create_annotated_csv(train_input, ann_path)
        print(f"Created {ann_path} ({count} rows)")
    else:
        print(f"Warning: {train_input} not found")

    test_input = data_dir / "test.txt"
    if test_input.exists():
        print("Processing test split...")
        basic_path = data_dir / "dnd.test.csv"
        count = create_basic_csv(test_input, basic_path)
        print(f"Created {basic_path} ({count} rows)")

        ann_path = data_dir / "dnd_ann.test.csv"
        count = create_annotated_csv(test_input, ann_path)
        print(f"Created {ann_path} ({count} rows)")
    else:
        print(f"Warning: {test_input} not found")

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Process DND dataset")
    parser.parse_args()

    data_dir = Path(__file__).parent.parent / "dnd"
    process_dnd_dataset(data_dir)


if __name__ == "__main__":
    main()
