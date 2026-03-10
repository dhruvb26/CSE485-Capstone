import logging

import torch

log = logging.getLogger(__name__)


def check_gpu() -> dict:
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_map = {i: torch.cuda.get_device_name(i) for i in range(num_gpus)}
        if num_gpus == 1:
            log.info("GPU is available and running on GPU %s", gpu_map[0])
        else:
            log.info(
                "Multiple GPUs detected: %s", "; ".join([f"{idx}: {name}" for idx, name in gpu_map.items()])
            )
        return gpu_map
    else:
        log.info("GPU is not available, exiting program.")
        exit(1)
