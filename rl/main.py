"""
Stage-0 evaluation harness.

Usage (from project root):
    python -m rl.main

Log files are written in SysEval-compatible format:
    runs/{timestamp}/{dataset}/{task_id}/{dataset}_{model_id}_{task_id}_{n}.json

To evaluate a completed run offline:
    from rl.evaluate import print_run_summary
    print_run_summary("runs/<timestamp>")
"""

import json
import logging
from pathlib import Path

from rl.evaluate import print_run_summary
from rl.handlers import TASK_REGISTRY, CasinoDatasetHandler, DNDDatasetHandler
from rl.helpers import create_run_dir
from rl.models import get_model

CONFIG_PATH = Path("rl/config.json")

logger = logging.getLogger(__name__)


def run(config: dict) -> dict:
    base_dir = Path(config["data"]["base_dir"])
    n_instances: int = config["eval"]["n_instances"]
    task_ids: list[str] = [
        tid for tasks in config["eval"]["tasks"].values() for tid in tasks
    ]

    model = get_model(config)
    run_dir = create_run_dir(config)
    logger.info("run dir: %s  model: %s", run_dir, model.model_id)

    ca_handler = CasinoDatasetHandler(base_dir / config["data"]["casino"]["test"])
    dnd_handler = DNDDatasetHandler(base_dir / config["data"]["dnd"]["test"])

    all_results: dict[str, dict] = {}

    for task_id in task_ids:
        if task_id not in TASK_REGISTRY:
            logger.warning("unknown task: %s", task_id)
            continue

        task = TASK_REGISTRY[task_id]()
        dataset = ca_handler if task_id.endswith("_ca") else dnd_handler
        agent = "mturk_agent_1" if task_id.endswith("_ca") else "YOU"

        result = task.evaluate(
            dataset, model, n=n_instances, agent=agent, run_dir=run_dir
        )
        all_results[task_id] = result
        logger.info("%s: %.2f%%", task_id, result["accuracy"] * 100)

    print_run_summary(run_dir)

    return all_results


if __name__ == "__main__":
    config = json.loads(CONFIG_PATH.read_text())
    logging.basicConfig(
        level=config["logging"]["level"],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    all_results = run(config)
