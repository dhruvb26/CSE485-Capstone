import fnmatch
import json
from pathlib import Path

import requests
from tqdm.auto import tqdm


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
