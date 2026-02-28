import logging

from rl.config import TrainConfig, load_config
from rl.evaluate import print_run_summary
from rl.handlers import TASK_REGISTRY, CasinoDatasetHandler, DNDDatasetHandler
from rl.helpers import create_run_dir
from rl.models import get_model

logger = logging.getLogger(__name__)


def run(cfg: TrainConfig) -> dict:
    base_dir = cfg.data.base_dir
    n_instances = cfg.eval.n_instances
    task_ids: list[str] = [
        tid for tasks in cfg.eval.tasks.values() for tid in tasks
    ]

    model = get_model(cfg)
    run_dir = create_run_dir(cfg)
    logger.info("run dir: %s  model: %s", run_dir, model.model_id)

    ca_handler = CasinoDatasetHandler(f"{base_dir}/{cfg.data.casino.test}")
    dnd_handler = DNDDatasetHandler(f"{base_dir}/{cfg.data.dnd.test}")

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
    cfg = load_config()
    logging.basicConfig(
        level=cfg.logging.level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(cfg)
