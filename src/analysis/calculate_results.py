# imports
import argparse
import json
import os

import numpy as np

LANGS = ["en", "fr", "ge", "it", "po", "ru", "chinese"]
SEEDS = [555, 666, 777]


def get_micro_f1(lang, seed, setting, base_dir):
    """Get f1-micro given langauge, seed, and experimental setting.

    Args:
        lang (str): langauge to be calculated.
        seed (int): specific seed in question.
        setting (str): which portion of the dataset used.
        base_dir (str): base directory of the results.

    Returns:
        float: f1-micro score.
    """
    file_path = f"{base_dir}/seed_{seed}/{setting}/results.txt"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r") as file:
        for line in file:
            result_dict = json.loads(line.strip())
            if result_dict.get("lang") == lang:
                return result_dict["test_f1_micro"]


# arg parser
def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate mean and standard deviation of F1-micro scores."
    )
    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        help="Dataset setting to evaluate ('chinese_only', 'sem_only', 'all').",
    )
    parser.add_argument(
        "--base_dir", type=str, required=True, help="Base directory containing results."
    )
    return parser.parse_args()


def main():
    # parse args
    args = parse_args()
    setting = args.setting
    base_dir = args.base_dir

    # get input numbers for each langauge
    results_dict = {}
    for lang in LANGS:
        results_dict["lang"] = []
        for seed in SEEDS:
            results_dict["lang"].append(get_micro_f1(lang, seed, setting, base_dir))
        print(f"Results for {lang}:")
        print(
            f"Mean: {np.mean(results_dict['lang'])}\tStd: {np.std(results_dict['lang'])}"
        )


if __name__ == "__main__":
    main()
