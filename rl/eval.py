import logging
from pathlib import Path

from rl.config import TrainConfig, load_config
from rl.evaluate import print_run_summary
from rl.handlers import (
    TASK_REGISTRY,
    CasinoDatasetHandler,
    DNDDatasetHandler,
)
from rl.helpers import create_run_dir
from rl.models import get_model

logger = logging.getLogger(__name__)

_SUFFIX_TO_DATASET = {"ca": "casino", "dnd": "dnd"}
_SUFFIX_TO_AGENT = {"ca": "mturk_agent_1", "dnd": "YOU"}


def run(cfg: TrainConfig) -> dict:
    base_dir = cfg.data.base_dir
    n_instances = cfg.eval.n_instances
    task_ids: list[str] = [tid for tasks in cfg.eval.tasks.values() for tid in tasks]

    model = get_model(cfg)
    run_dir = create_run_dir(cfg, Path("runs"))
    logger.info("run dir: %s  model: %s", run_dir, model.model_id)

    handlers: dict[str, object] = {}
    if hasattr(cfg.data, "casino") and cfg.data.casino:
        handlers["ca"] = CasinoDatasetHandler(f"{base_dir}/{cfg.data.casino.test}")
    if hasattr(cfg.data, "dnd") and cfg.data.dnd:
        handlers["dnd"] = DNDDatasetHandler(f"{base_dir}/{cfg.data.dnd.test}")

    all_results: dict[str, dict] = {}

    for task_id in task_ids:
        if task_id not in TASK_REGISTRY:
            logger.warning("unknown task: %s", task_id)
            continue

        suffix = task_id.rsplit("_", 1)[-1]
        handler = handlers.get(suffix)
        if handler is None:
            logger.warning(
                "no dataset handler for task %s (suffix=%s)", task_id, suffix
            )
            continue

        agent = _SUFFIX_TO_AGENT.get(suffix, "mturk_agent_1")
        task = TASK_REGISTRY[task_id]()

        result = task.evaluate(
            handler, model, n=n_instances, agent=agent, run_dir=run_dir
        )
        all_results[task_id] = result
        logger.info("%s: %.2f%%", task_id, result["accuracy"] * 100)

    print_run_summary(run_dir)

    return all_results


if __name__ == "__main__":
    cfg = load_config(Path("rl/configs/eval.yaml"))
    logging.basicConfig(
        level=cfg.logging.level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(cfg)
