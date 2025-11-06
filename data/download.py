import argparse
import logging
import os
import shutil
from urllib.parse import urljoin

import requests
import urllib3
from convokit import Corpus, download

# Disable SSL warnings for expired certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

DATASETS = {
    "casino": {
        "name": "casino-corpus",
        "description": "The CaSiNo corpus is a collection of conversations between buyers and sellers in a casino. It is a small dataset that is used to test the performance of the bargaining agents.",
        "url": "https://convokit.cornell.edu/documentation/casino-corpus.html#:~:text=Data%20License-,CaSiNo%20Corpus,Association%20for%20Computational%20Linguistics.",
    },
    "amazon_history_price": {
        "name": "amazon-price-history",
        "description": "Amazon price history dataset containing historical pricing data for various product categories from Amazon.",
        "url": "https://github.com/TianXiaSJTU/AmazonPriceHistory",
        "base_url": "https://raw.githubusercontent.com/TianXiaSJTU/AmazonPriceHistory/main/data/AmazonHistoryPrice/",
        "files": [
            "automotive.json",
            "baby-products.json",
            "beauty.json",
            "books.json",
            "electronics.json",
            "health-personal-care.json",
            "home-kitchen.json",
            "industrial-scientific.json",
            "movies-tv.json",
            "music.json",
            "other.json",
            "patio-lawn-garden.json",
            "pet-supplies.json",
            "software.json",
            "sports-outdoors.json",
            "tools-home-improvement.json",
            "toys-games.json",
            "video-games.json",
        ],
    },
    "craigslist_bargains": {
        "name": "craigslist-bargains",
        "description": "Craigslist bargains dataset containing negotiation dialogues between buyers and sellers from Stanford NLP.",
        "url": "https://huggingface.co/datasets/stanfordnlp/craigslist_bargains",
        "files_urls": {
            "train.json": "https://worksheets.codalab.org/rest/bundles/0xd34bbbc5fb3b4fccbd19e10756ca8dd7/contents/blob/parsed.json",
            "validation.json": "https://worksheets.codalab.org/rest/bundles/0x15c4160b43d44ee3a8386cca98da138c/contents/blob/parsed.json",
            "test.json": "https://worksheets.codalab.org/rest/bundles/0x54d325bbcfb2463583995725ed8ca42b/contents/blob/",
        },
    },
}


def _download_github_files(base_url: str, files: list, output_dir: str):
    """Download files from GitHub repository using raw URLs.

    Args:
        base_url: Base GitHub raw URL for the files
        files: List of filenames to download
        output_dir: Directory to save downloaded files
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in files:
        file_url = urljoin(base_url, filename)
        output_path = os.path.join(output_dir, filename)

        logger.info(f"Downloading {filename}...")
        try:
            response = requests.get(file_url, timeout=30)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)

            logger.info(f"✓ Downloaded {filename}")

        except requests.RequestException as e:
            logger.error(f"✗ Failed to download {filename}: {e}")
            raise


def _download_files_from_urls(files_urls: dict, output_dir: str):
    """Download files from a mapping of filenames to URLs.

    Args:
        files_urls: Dictionary mapping filenames to their download URLs
        output_dir: Directory to save downloaded files
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename, url in files_urls.items():
        output_path = os.path.join(output_dir, filename)

        logger.info(f"Downloading {filename} from {url}...")
        try:
            # Disable SSL verification for codalab.org URLs with expired certificates
            response = requests.get(url, timeout=60, verify=False)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)

            logger.info(f"✓ Downloaded {filename}")

        except requests.RequestException as e:
            logger.error(f"✗ Failed to download {filename}: {e}")
            raise


def download_dataset(dataset_name: str, overwrite: bool = False):
    """Download and save a dataset to data/{dataset_name} directory.

    Args:
        dataset_name: Name of the dataset to download
        overwrite: If True, overwrite existing data. If False, skip if data exists.
    """

    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. Available: {list(DATASETS.keys())}"
        )

    dataset_info = DATASETS[dataset_name]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, dataset_name)

    logger.info(f"Output directory: {output_dir}")

    if os.path.exists(output_dir) and os.listdir(output_dir):
        if not overwrite:
            logger.info(
                f"✓ Dataset '{dataset_name}' already exists at {output_dir}\n"
                f"  Use --overwrite to download and replace existing data."
            )
            return
        else:
            logger.info(f"Removing existing data at {output_dir}...")
            shutil.rmtree(output_dir)

    if dataset_name == "casino":
        logger.info(f"Downloading {dataset_name} corpus...")

        cache_dir = os.path.join(script_dir, ".cache")
        path = download(dataset_info["name"], data_dir=cache_dir)

        logger.info(f"Loading corpus from {path}...")
        corpus = Corpus(filename=path)

        logger.info(f"Saving to {output_dir}...")
        corpus.dump(output_dir)

        logger.info(f"✓ Dataset '{dataset_name}' downloaded and saved to {output_dir}")
    elif dataset_name == "amazon_history_price":
        logger.info(f"Downloading {dataset_name} dataset...")

        base_url = dataset_info["base_url"]
        files = dataset_info["files"]

        _download_github_files(base_url, files, output_dir)

        logger.info(f"✓ Dataset '{dataset_name}' downloaded and saved to {output_dir}")
    elif dataset_name == "craigslist_bargains":
        logger.info(f"Downloading {dataset_name} dataset...")

        files_urls = dataset_info["files_urls"]

        _download_files_from_urls(files_urls, output_dir)

        logger.info(f"✓ Dataset '{dataset_name}' downloaded and saved to {output_dir}")
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASETS.keys()),
        help="Dataset to download",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing data if it already exists",
    )

    args = parser.parse_args()
    download_dataset(args.dataset, overwrite=args.overwrite)
