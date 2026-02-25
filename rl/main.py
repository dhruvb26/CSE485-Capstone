"""
Stage-0 evaluation harness.

Usage (from project root):
    python -m rl.main
"""

import json
from pathlib import Path

from rl.handlers import TASK_REGISTRY, CasinoDatasetHandler, DNDDatasetHandler
from rl.models import get_model

CONFIG_PATH = Path("rl/config.json")


def run(config: dict) -> dict:
    base_dir = Path(config["data"]["base_dir"])
    n_instances: int = config["eval"]["n_instances"]
    task_ids: list[str] = [
        tid for tasks in config["eval"]["tasks"].values() for tid in tasks
    ]

    model = get_model(config)

    ca_handler = CasinoDatasetHandler(base_dir / config["data"]["casino"]["test"])
    dnd_handler = DNDDatasetHandler(base_dir / config["data"]["dnd"]["test"])

    all_results: dict[str, dict] = {}

    for task_id in task_ids:
        if task_id not in TASK_REGISTRY:
            print(f"unknown task: {task_id}")
            continue

        task = TASK_REGISTRY[task_id]()
        dataset = ca_handler if task_id.endswith("_ca") else dnd_handler
        agent = "mturk_agent_1" if task_id.endswith("_ca") else "YOU"

        results = task.evaluate(dataset, model, n=n_instances, agent=agent)
        all_results[task_id] = results
        print(f"{task_id}: {results['accuracy']:.2%}")

    return all_results


if __name__ == "__main__":
    config = json.loads(CONFIG_PATH.read_text())
    all_results = run(config)
