"""
Script to convert DND (Deal or No Deal) dataset files into CSV format.

Input files:
- dnd/train.txt: Training data with tagged format
- dnd/test.txt: Test data with tagged format

Output files:
1. Basic DND format:
   - dnd/dnd.train.csv: Training data with input, dialogue, output, partner_input
   - dnd/dnd.test.csv: Test data with same format

2. Annotated DND format:
   - dnd/dnd_ann.train.csv: Training data with uuid, scenario, agents_info, etc.
   - dnd/dnd_ann.test.csv: Test data with same format

Usage:
    python dnd.py              # Process dataset and create all CSV files
    python dnd.py --example    # Extract first row to example JSON files
"""

import argparse
import csv
import json
import re
import uuid
from pathlib import Path


def parse_tagged_line(line: str) -> dict | None:
    """
    Parse a line from train.txt/test.txt with the tagged format:
    <input> ... </input> <dialogue> ... </dialogue> <output> ... </output> <partner_input> ... </partner_input>
    """
    # Extract tagged sections using regex
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
    """
    Parse input string "1 4 4 1 1 2" into structured format.
    Format: count_book value_book count_hat value_hat count_ball value_ball
    Returns: {"count": [books, hats, balls], "value": [book_val, hat_val, ball_val]}
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
    """
    Parse output string "item0=0 item1=4 item2=0 item0=1 item1=0 item2=1" into allocations.
    First 3 items = YOU's allocation, last 3 items = THEM's allocation.

    Handle disagree case: "<disagree> <disagree> ..."
    """
    if "<disagree>" in output_str or "no agreement" in output_str.lower():
        return None, None

    # Parse item assignments
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
    """
    Create basic DND CSV with columns: input, dialogue, output, partner_input
    Returns number of rows processed.
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

            # Convert input values to JSON format
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

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["input", "dialogue", "output", "partner_input"]
        )
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def create_annotated_csv(input_path: Path, output_path: Path) -> int:
    """
    Create annotated DND CSV with columns: uuid, scenario, agents_info, scenario_uuid, agents, outcome, events

    Note: Since we don't have annotation metadata (intent, proposal, etc.),
    events will have basic structure without detailed annotations.
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

            # Parse values
            you_input = parse_input_values(parsed["input"])
            them_input = parse_input_values(parsed["partner_input"])
            you_alloc, them_alloc = parse_output_allocation(parsed["output"])

            # Generate UUIDs
            nego_uuid = f"E_{uuid.uuid4().hex[:16]}"
            scenario_uuid = f"FB_{uuid.uuid4().hex[:16]}"

            # Build scenario
            scenario = {
                "attributes": [],
                "uuid": scenario_uuid,
                "kbs": [
                    # Agent 0 (YOU) knowledge base
                    [
                        {
                            "Count": you_input["count"][0],
                            "Name": "book",
                            "Value": you_input["value"][0],
                        },
                        {
                            "Count": you_input["count"][1],
                            "Name": "hat",
                            "Value": you_input["value"][1],
                        },
                        {
                            "Count": you_input["count"][2],
                            "Name": "ball",
                            "Value": you_input["value"][2],
                        },
                    ],
                    # Agent 1 (THEM) knowledge base
                    [
                        {
                            "Count": them_input["count"][0],
                            "Name": "book",
                            "Value": them_input["value"][0],
                        },
                        {
                            "Count": them_input["count"][1],
                            "Name": "hat",
                            "Value": them_input["value"][1],
                        },
                        {
                            "Count": them_input["count"][2],
                            "Name": "ball",
                            "Value": them_input["value"][2],
                        },
                    ],
                ],
            }

            # Build outcome
            valid_deal = you_alloc is not None
            if valid_deal:
                # Calculate rewards
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

            # Build events from dialogue
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

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "uuid",
                "scenario",
                "agents_info",
                "scenario_uuid",
                "agents",
                "outcome",
                "events",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def parse_dialogue_to_events(dialogue: str, final_alloc: dict | None) -> list[dict]:
    """
    Parse dialogue string into events list.

    Dialogue format: "YOU: message <eos> THEM: message <eos> ... <selection>"

    Since we don't have annotation metadata, we'll create basic event structures.
    """
    events = []

    # Split by <eos>
    turns = dialogue.split("<eos>")

    for i, turn in enumerate(turns):
        turn = turn.strip()
        if not turn:
            continue

        # Check if this is a selection turn
        if "<selection>" in turn:
            # Extract speaker and handle selection
            if turn.startswith("YOU:"):
                agent = 0
                remaining = turn[4:].strip()
            elif turn.startswith("THEM:"):
                agent = 1
                remaining = turn[5:].strip()
            else:
                # Just <selection> without speaker prefix
                if "YOU:" in turns[i - 1] if i > 0 else False:
                    agent = 1  # THEM made selection after YOU's message
                else:
                    agent = 0
                remaining = turn

            # Add message event if there's text before <selection>
            text_before = remaining.replace("<selection>", "").strip()
            if text_before:
                events.append(
                    {
                        "agent": agent,
                        "time": len(events),
                        "action": "message",
                        "data": text_before,
                        "metadata": {
                            "intent": "unknown",
                            "proposal": None,
                            "proposal_type": None,
                        },
                    }
                )

            # Add selection event
            if final_alloc:
                events.append(
                    {
                        "agent": agent,
                        "time": len(events),
                        "action": "select",
                        "data": final_alloc if agent == 0 else None,
                        "metadata": {"intent": "select"},
                    }
                )
        else:
            # Regular message
            if turn.startswith("YOU:"):
                agent = 0
                text = turn[4:].strip()
            elif turn.startswith("THEM:"):
                agent = 1
                text = turn[5:].strip()
            else:
                continue

            # Infer basic intent
            intent = infer_basic_intent(text)

            events.append(
                {
                    "agent": agent,
                    "time": len(events),
                    "action": "message",
                    "data": text,
                    "metadata": {
                        "intent": intent,
                        "proposal": None,
                        "proposal_type": None,
                    },
                }
            )

    return events


def infer_basic_intent(text: str) -> str:
    """
    Infer basic dialogue act from text.
    This is a simple heuristic - actual annotations would be more accurate.
    """
    text_lower = text.lower()

    if any(
        word in text_lower
        for word in ["deal", "ok", "okay", "yes", "agree", "sounds good", "sure"]
    ):
        if any(word in text_lower for word in ["no deal", "no agreement"]):
            return "disagree"
        return "agree"

    if any(
        word in text_lower
        for word in ["no", "can't", "won't", "cannot", "not going to", "not possible"]
    ):
        return "disagree"

    if "?" in text:
        return "inquire"

    if any(word in text_lower for word in ["hi", "hello", "hey"]):
        return "greet"

    if any(
        word in text_lower
        for word in [
            "i'd like",
            "i want",
            "i need",
            "i'll take",
            "how about",
            "give me",
            "can i have",
        ]
    ):
        return "propose"

    return "unknown"


def extract_examples(csv_dir: Path, examples_dir: Path):
    """Extract first row from each CSV file to example JSON files."""
    examples_dir.mkdir(parents=True, exist_ok=True)

    # Basic CSV example
    basic_csv = csv_dir / "dnd.train.csv"
    if basic_csv.exists():
        with open(basic_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        example = {
            "input": json.loads(row["input"]),
            "dialogue": row["dialogue"],
            "output": row["output"],
            "partner_input": json.loads(row["partner_input"]),
        }

        with open(examples_dir / "basic_example.json", "w", encoding="utf-8") as f:
            json.dump(example, f, indent=2)
        print(f"Saved {examples_dir / 'basic_example.json'}")

    # Annotated CSV example
    ann_csv = csv_dir / "dnd_ann.train.csv"
    if ann_csv.exists():
        with open(ann_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        example = {
            "uuid": row["uuid"],
            "scenario": json.loads(row["scenario"]),
            "agents_info": json.loads(row["agents_info"]),
            "scenario_uuid": row["scenario_uuid"],
            "agents": json.loads(row["agents"]),
            "outcome": json.loads(row["outcome"]),
            "events": json.loads(row["events"]),
        }

        with open(examples_dir / "annotated_example.json", "w", encoding="utf-8") as f:
            json.dump(example, f, indent=2)
        print(f"Saved {examples_dir / 'annotated_example.json'}")

    print(f"\nExtracted examples to {examples_dir}")


def process_dnd_dataset(data_dir: Path):
    """Process all DND dataset files."""

    # Process train.txt
    train_input = data_dir / "train.txt"
    if train_input.exists():
        # Basic format
        basic_train = data_dir / "dnd.train.csv"
        count = create_basic_csv(train_input, basic_train)
        print(f"Created {basic_train} with {count} rows")

        # Annotated format
        ann_train = data_dir / "dnd_ann.train.csv"
        count = create_annotated_csv(train_input, ann_train)
        print(f"Created {ann_train} with {count} rows")
    else:
        print(f"Warning: {train_input} not found")

    # Process test.txt
    test_input = data_dir / "test.txt"
    if test_input.exists():
        # Basic format
        basic_test = data_dir / "dnd.test.csv"
        count = create_basic_csv(test_input, basic_test)
        print(f"Created {basic_test} with {count} rows")

        # Annotated format
        ann_test = data_dir / "dnd_ann.test.csv"
        count = create_annotated_csv(test_input, ann_test)
        print(f"Created {ann_test} with {count} rows")
    else:
        print(f"Warning: {test_input} not found")

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Process DND dataset or extract example data"
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Extract first row from CSVs to example JSON files",
    )

    args = parser.parse_args()

    # Set paths relative to this script
    script_dir = Path(__file__).parent
    data_dir = script_dir / "dnd"
    examples_dir = data_dir / "examples"

    if args.example:
        basic_csv = data_dir / "dnd.train.csv"
        if not basic_csv.exists():
            print("CSV files not found. Run without --example first to generate them.")
            return
        extract_examples(data_dir, examples_dir)
    else:
        process_dnd_dataset(data_dir)


if __name__ == "__main__":
    main()
