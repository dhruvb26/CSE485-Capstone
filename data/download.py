import argparse
import logging
import os
import shutil

from convokit import Corpus, download

logger = logging.getLogger(__name__)

DATASETS = {
    "casino": {
        "name": "casino-corpus",
        "description": "The CaSiNo corpus is a collection of conversations between buyers and sellers in a casino. It is a small dataset that is used to test the performance of the bargaining agents.",
        "url": "https://convokit.cornell.edu/documentation/casino-corpus.html#:~:text=Data%20License-,CaSiNo%20Corpus,Association%20for%20Computational%20Linguistics.",
        "license": "CC-BY-NC-SA 4.0",
    },
}


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
