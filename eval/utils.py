"""Utility functions for the evaluation framework."""

import importlib
import json
import os
import csv


def dynamic_import(dotted_path):
    """Import a class from a dotted path like 'eval.tasks.sta_total_item_count_dnd.TICNDHandlerDND'."""
    module_name, _, class_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_output_path(storage_dir, dataset_name, model_name, task_name, num_instances, args=None):
    """Construct the output path for evaluation logs."""
    fname = f"{dataset_name}_{model_name}_{task_name}_{num_instances}"

    if args is not None:
        if getattr(args, "num_utts_partial_dial", -1) != -1:
            fname += f"_partial_{args.num_utts_partial_dial}"
        if getattr(args, "use_cot", False):
            fname += "_cot"
        if getattr(args, "num_multishot", 0) != 0:
            fname += f"_multishot_{args.num_multishot}"
        if getattr(args, "num_prior_utts", 0) != 0:
            fname += f"_prior_utts_{args.num_prior_utts}"

    fname = f"{fname}.json"
    return os.path.join(storage_dir, fname)


def json_loader(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def write_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for line in data:
            json.dump(line, f)
            f.write("\n")


def write_to_csv(data, file_path):
    assert isinstance(data, list), "data must be a list of strings"
    with open(file_path, "w") as f:
        write = csv.writer(f)
        write.writerows(data)
