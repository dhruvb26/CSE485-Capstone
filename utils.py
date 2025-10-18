import atexit
import glob
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from typing import Dict, Optional, Tuple

import requests
import torch

logger = logging.getLogger(__name__)

_vllm_process = None


def check_gpu() -> None:
    """Check whether a GPU is available, exit if not."""
    if torch.cuda.is_available():
        logger.info(
            "GPU is available and running on GPU %s", torch.cuda.get_device_name(0)
        )
    else:
        logger.info("GPU is not available, exiting program.")
        exit(1)


def _cleanup_subprocess(signum=None, frame=None):
    """Clean up the vLLM subprocess on exit or interrupt."""
    global _vllm_process
    if _vllm_process is not None:
        logger.info("Shutting down vLLM server subprocess...")
        try:
            # Try to terminate the entire process group (covers child processes spawned by vLLM)
            try:
                pgid = os.getpgid(_vllm_process.pid)
                os.killpg(pgid, signal.SIGTERM)
            except Exception:
                # Fallback to terminating the single process if process groups are unavailable
                _vllm_process.terminate()

            # Wait up to 5 seconds for graceful termination
            try:
                _vllm_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate gracefully
                logger.warning("Process didn't terminate gracefully, force killing.")
                try:
                    pgid = os.getpgid(_vllm_process.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except Exception:
                    _vllm_process.kill()
                _vllm_process.wait()
            logger.info("vLLM server subprocess terminated successfully.")
        except Exception as e:
            logger.error(f"Error while terminating subprocess: {e}")
        finally:
            _vllm_process = None

    # If called from signal handler, exit the program
    if signum is not None:
        sys.exit(0)


def start_vllm_server(
    model: str,
    port: int,
    host: str,
    dtype: str,
    tensor_parallel_size: int,
    cache_dir: str,
):
    """Launch vLLM's OpenAI-compatible server."""
    global _vllm_process

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--dtype",
        dtype,
        "--port",
        str(port),
        "--host",
        host,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
    ]

    logger.info(f"Starting vLLM server with model: {model}")
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Serving at: http://{host}:{port}\n")

    env = os.environ.copy()
    env["HF_HOME"] = cache_dir

    logger.info("Starting vLLM server process.")
    process = subprocess.Popen(
        cmd, env=env, stdout=sys.stdout, stderr=sys.stderr, start_new_session=True
    )

    _vllm_process = process
    signal.signal(signal.SIGINT, _cleanup_subprocess)
    signal.signal(signal.SIGTERM, _cleanup_subprocess)
    atexit.register(_cleanup_subprocess)

    client_host = "localhost" if host in ("0.0.0.0", "::") else host
    ready_url = f"http://{client_host}:{port}/v1/models"

    logger.info(f"Waiting for server readiness at {ready_url}")

    for attempt in range(500):
        if process.poll() is not None:
            logger.error("vLLM server process has terminated unexpectedly")
            logger.error("Check the error messages above for details")
            raise RuntimeError("vLLM server exited before becoming ready")

        try:
            response = requests.get(ready_url, timeout=5)
            if response.ok:
                logger.info("=" * 80)
                logger.info("✓ vLLM server is ready!")
                logger.info("=" * 80)
                return process
        except requests.exceptions.RequestException:
            pass

        if attempt % 15 == 0 and attempt > 0:
            elapsed = attempt * 2
            logger.info(f"Still waiting for server... ({elapsed}s elapsed)")
        time.sleep(2)

    logger.error("vLLM server did not start within the expected time window")
    raise TimeoutError("Timeout waiting for vLLM server readiness")


def load_product(dataset_dir: str, product_index: int = 0) -> Dict:
    """
    Load a product by global index across all JSON files in dataset_dir.

    This version automatically moves to the next JSON file when the index
    exceeds the number of items in the current file.
    """
    files = sorted(glob.glob(os.path.join(dataset_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {dataset_dir}")

    # Load each file sequentially until the correct product_index is reached
    remaining = product_index
    for file_path in files:
        with open(file_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0:
            continue  # skip empty files

        if remaining < len(data):
            item = data[remaining]
            filename = os.path.splitext(os.path.basename(file_path))[0]
            codename = f"{filename}_{remaining}"

            def parse_price(p):
                if isinstance(p, (int, float)):
                    return float(p)
                if isinstance(p, str):
                    return float(p.replace("$", "").replace(",", ""))
                return 0.0

            highest = parse_price(item.get("highest_price"))
            lowest = parse_price(item.get("lowest_price"))

            return {
                "title": item.get("title"),
                "description": item.get("description"),
                "highest_price": highest,
                "lowest_price": lowest,
                "category": item.get("category"),
                "codename": codename,
            }
        else:
            # Move to next file, adjust remaining index
            remaining -= len(data)

    # If we get here, index exceeded total available items
    raise IndexError(
        f"Product index {product_index} exceeds total products across all JSON files."
    )


def inventory_list(item: dict) -> str:
    return (
        "Inventory List:\n"
        f"Product (codename: {item['codename']})\n"
        f'Title: "{item["title"]}"\n'
        f'Description: "{item["description"]}"\n'
        f"Listing Price: ${item['highest_price']:.2f} per item\n"
        "Available Quantity: 1\n"
    )


def shopping_list(item: dict, budget: float) -> str:
    """Generate shopping list for buyer with budget information."""
    return (
        "Shopping List\n"
        f"codename: {item['codename']}\n"
        f"quantity: 1\n"
        f"budget: ${budget:.2f}"
    )


_ACTION_RE = re.compile(r"\[(BUY|SELL|DEAL|REJECT|QUIT)\]", re.I)
_PRICE_RE = re.compile(r"\$?\s*([0-9,]+(?:\.[0-9]{1,2})?)")
_ACTION_LINE_RE = re.compile(
    r"^(?:\s*Action:\s*)?\[(BUY|SELL|DEAL|REJECT|QUIT)\]\s*(?:\$\s*([0-9,]+(?:\.[0-9]{1,2})?))?",
    re.I,
)


def extract_action_and_price(text: str) -> Tuple[str, Optional[float]]:
    """
    Extract the action and price from the *action line only*.
    We scan lines bottom-up and return the first line that looks like an action.
    Price is only taken if it appears on the same action line.
    Handles prices with commas (e.g., $1,000 -> 1000.0).
    """
    if not text:
        return "INVALID", None

    action, price = "INVALID", None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        m = _ACTION_LINE_RE.match(ln)
        if not m:
            continue
        action = m.group(1).upper()
        if action in {"BUY", "SELL", "DEAL"}:
            p = m.group(2)
            # Remove commas from price string before converting to float
            price = float(p.replace(",", "")) if p is not None else None
        else:
            price = None
        return action, price
    return action, price


def compute_metrics(outcome: Dict, B: float, C: float) -> Dict:
    """
    Per-session metrics (profit and normalized profit) + MI/CI split.
    """
    res = {
        "valid": not outcome.get("invalid", False),
        "deal": outcome.get("deal", False),
        "scenario": "MI" if C <= B else "CI",
    }
    if res["deal"]:
        D = outcome["price"]
        denom = abs(B - C) if abs(B - C) > 1e-9 else 1.0
        res["Pb"] = B - D
        res["Ps"] = D - C
        res["NPb"] = (B - D) / denom
        res["NPs"] = (D - C) / denom
    else:
        res["Pb"] = res["Ps"] = res["NPb"] = res["NPs"] = 0.0
    return res


def accumulate(agg: Dict, one: Dict) -> None:
    """Accumulate metrics into ALL and MI/CI splits."""
    for split in ["ALL", one["scenario"]]:
        a = agg.setdefault(
            split,
            {
                "n": 0,
                "valid": 0,
                "deal": 0,
                "SPb": 0.0,
                "SPs": 0.0,
                "SNPb": 0.0,
                "SNPs": 0.0,
            },
        )
        a["n"] += 1
        a["valid"] += int(one["valid"])
        a["deal"] += int(one["deal"])
        a["SPb"] += one["Pb"]
        a["SPs"] += one["Ps"]
        a["SNPb"] += one["NPb"]
        a["SNPs"] += one["NPs"]


def finalize_aggregates(agg: Dict) -> Dict:
    """Compute final summary numbers for each split."""
    out = {}
    for split, a in agg.items():
        n = max(1, a["n"])
        out[split] = {
            "n": a["n"],
            "valid_rate": a["valid"] / n,
            "deal_rate": a["deal"] / n,
            "SP_b": a["SPb"],
            "SP_s": a["SPs"],
            "SNP_b": a["SNPb"],
            "SNP_s": a["SNPs"],
        }
    return out
