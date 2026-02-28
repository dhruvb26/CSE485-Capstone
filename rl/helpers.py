import fnmatch
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm.auto import tqdm

from rl.config import TrainConfig


def create_run_dir(cfg: TrainConfig, runs_dir: Path | None = None) -> Path:
    if runs_dir is None:
        runs_dir = Path.cwd() / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc)
    run_id = ts.strftime("%Y-%m-%dT%H-%M-%S.") + ts.strftime("%f")[:3] + "Z"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return run_dir


def download_data(out_dir: Path = Path("data")) -> list[Path]:
    """Download data from the GitHub repository and return the list of paths to the downloaded files."""
    OWNER = "dhruvb26"
    REPO = "SysEval-NegoLLMs"
    PATH_IN_REPO = "storage/utilities"
    BRANCH = "main"

    out_dir.mkdir(parents=True, exist_ok=True)

    api_url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{PATH_IN_REPO}?ref={BRANCH}"
    response = requests.get(api_url)
    response.raise_for_status()
    items = response.json()

    def keep(name: str) -> bool:
        return name.endswith(".csv") and (
            fnmatch.fnmatch(name, "dnd*") or fnmatch.fnmatch(name, "ca*")
        )

    csv_files = [
        it for it in items if it.get("type") == "file" and keep(it.get("name", ""))
    ]

    print(f"Found {len(csv_files)} matching CSV files.")

    downloaded_paths = []

    for it in tqdm(csv_files, desc="Fetching"):
        url = it["download_url"]
        out_path = out_dir / it["name"]
        r = requests.get(url)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        downloaded_paths.append(out_path)
    return downloaded_paths
