import argparse
import gzip
import os
import shutil
import zipfile
from urllib.parse import urljoin

import requests
import urllib3
from convokit import Corpus, download

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
    "ebay_best_offer": {
        "name": "eBay Best Offer Bargaining Data",
        "description": "eBay Best Offer Bargaining data containing a product, its listed price, and metadata as well as a buyer offer which in some cases leads to a sequential seller counteroffer and a deal.",
        "url": "https://www.nber.org/research/data/best-offer-sequential-bargaining",
        "files_urls": {
            "anon_bo_lists.csv.gz": "https://nber.org/bargaining/anon_bo_lists.csv.gz",
            "anon_bo_threads.csv.gz": "https://nber.org/bargaining/anon_bo_threads.csv.gz",
        },
    },
    "dnd": {
        "name": "Deal or No Deal Negotiation Dataset",
        "description": "Facebook Research Deal or No Deal dataset containing multi-issue bargaining dialogues where two agents negotiate to split a set of items (books, hats, balls) with different values.",
        "url": "https://github.com/facebookresearch/end-to-end-negotiator",
        "base_url": "https://raw.githubusercontent.com/facebookresearch/end-to-end-negotiator/master/src/data/negotiate/",
        "files": [
            "data.txt",
            "selfplay.txt",
            "test.txt",
            "train.txt",
            "val.txt",
        ],
    },
    "ji": {
        "name": "Job Interview Negotiation Dataset",
        "description": "Human-human negotiation dialogue dataset in a job interview scenario featuring dialogue act-based breakdown detection. From EACL 2021 paper.",
        "url": "https://github.com/gucci-j/negotiation-breakdown-detection",
        "zip_url": "https://raw.githubusercontent.com/gucci-j/negotiation-breakdown-detection/main/data.zip",
    },
}


def _download_github_files(base_url: str, files: list, output_dir: str):
    """Download files from a GitHub repository using raw URLs.

    Args:
        base_url: Base GitHub raw URL for the files.
        files: List of filenames to download.
        output_dir: Directory to save downloaded files.

    Raises:
        requests.RequestException: If any download fails.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in files:
        file_url = urljoin(base_url, filename)
        output_path = os.path.join(output_dir, filename)

        print(f"Downloading {filename}...")
        try:
            response = requests.get(file_url, timeout=30)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)

            print(f"Downloaded {filename}")

        except requests.RequestException as e:
            print(f"Error: Failed to download {filename}: {e}")
            raise


def _download_files_from_urls(files_urls: dict, output_dir: str):
    """Download files from a mapping of filenames to URLs.

    Args:
        files_urls: Dictionary mapping filenames to their download URLs.
        output_dir: Directory to save downloaded files.

    Raises:
        requests.RequestException: If any download fails.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename, url in files_urls.items():
        output_path = os.path.join(output_dir, filename)

        print(f"Downloading {filename} from {url}...")
        try:
            response = requests.get(url, timeout=60, verify=False)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)

            print(f"Downloaded {filename}")

        except requests.RequestException as e:
            print(f"Error: Failed to download {filename}: {e}")
            raise


def download_dataset(dataset_name: str, overwrite: bool = False):
    """Download and save a dataset to ``data/{dataset_name}/``.

    Dispatches to the appropriate download method based on the dataset
    name. Supports ConvoKit corpora, GitHub raw files, direct URL
    downloads, and ZIP/GZ extraction.

    Args:
        dataset_name: One of the keys in :data:`DATASETS`.
        overwrite: If True, remove and re-download existing data.
            Defaults to False (skip if directory already has files).

    Raises:
        ValueError: If the dataset name is not recognized.
        requests.RequestException: If any download fails.
    """
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. Available: {list(DATASETS.keys())}"
        )

    dataset_info = DATASETS[dataset_name]
    data_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(data_root, dataset_name)

    print(f"Output directory: {output_dir}")

    if os.path.exists(output_dir) and os.listdir(output_dir):
        if not overwrite:
            print(
                f"Dataset '{dataset_name}' already exists at {output_dir}\n"
                f"  Use --overwrite to download and replace existing data."
            )
            return
        else:
            print(f"Removing existing data at {output_dir}...")
            shutil.rmtree(output_dir)

    if dataset_name == "casino":
        print(f"Downloading {dataset_name} corpus...")

        cache_dir = os.path.join(data_root, ".cache")
        path = download(dataset_info["name"], data_dir=cache_dir)

        print(f"Loading corpus from {path}...")
        corpus = Corpus(filename=path)

        print(f"Saving to {output_dir}...")
        corpus.dump(output_dir)

        print(f"Created {output_dir}")
    elif dataset_name == "amazon_history_price":
        print(f"Downloading {dataset_name} dataset...")

        _download_github_files(dataset_info["base_url"], dataset_info["files"], output_dir)

        print(f"Created {output_dir}")
    elif dataset_name == "craigslist_bargains":
        print(f"Downloading {dataset_name} dataset...")

        _download_files_from_urls(dataset_info["files_urls"], output_dir)

        print(f"Created {output_dir}")
    elif dataset_name == "ebay_best_offer":
        print(f"Downloading {dataset_name} dataset...")

        _download_files_from_urls(dataset_info["files_urls"], output_dir)

        for file_name in dataset_info["files_urls"]:
            if file_name.endswith(".gz"):
                print(f"Unzipping {file_name}...")
                file_path = os.path.join(output_dir, file_name)
                ungz_path = os.path.join(output_dir, file_name[:-3])
                try:
                    with gzip.open(file_path, "rb") as file_in:
                        with open(ungz_path, "wb") as file_out:
                            shutil.copyfileobj(file_in, file_out)
                except Exception as e:
                    print(f"Error: Failed to unzip {file_name}: {e}")

    elif dataset_name == "dnd":
        print(f"Downloading {dataset_name} dataset...")

        _download_github_files(dataset_info["base_url"], dataset_info["files"], output_dir)

        print(f"Created {output_dir}")

    elif dataset_name == "ji":
        print(f"Downloading {dataset_name} dataset...")

        zip_url = dataset_info["zip_url"]
        os.makedirs(output_dir, exist_ok=True)

        zip_path = os.path.join(output_dir, "data.zip")
        print(f"Downloading data.zip from {zip_url}...")
        try:
            response = requests.get(zip_url, timeout=60)
            response.raise_for_status()

            with open(zip_path, "wb") as f:
                f.write(response.content)

            print("Downloaded data.zip")

        except requests.RequestException as e:
            print(f"Error: Failed to download data.zip: {e}")
            raise

        print("Extracting data.zip...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)

            os.remove(zip_path)
            print("Extracted and cleaned up data.zip")

        except zipfile.BadZipFile as e:
            print(f"Error: Failed to extract data.zip: {e}")
            raise

        print(f"Created {output_dir}")

    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")


if __name__ == "__main__":
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
