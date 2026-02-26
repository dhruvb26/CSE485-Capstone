import fnmatch
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm.auto import tqdm


def create_run_dir(config: dict, runs_dir: Path | None = None) -> Path:
    """Create a timestamped run directory, write config.json, return run_dir.

    Log files are written under run_dir / {dataset} / {task_id} /
    {dataset}_{model_id}_{task_id}_{n}.json in SysEval-compatible format.
    """
    if runs_dir is None:
        runs_dir = Path.cwd() / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc)
    run_id = ts.strftime("%Y-%m-%dT%H-%M-%S.") + ts.strftime("%f")[:3] + "Z"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return run_dir


def load_json(file_path: str) -> dict:
    """Load a JSON file into a dictionary."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None


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
