"""
Main class for the classification pipeline.
"""

import json
import os
import random

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import TrainerCallback

from config_parser import log_args, merge_config, parse_args
from preprocessing.label_generator import SemEvalTopicLabelGenerator

# Load the data
from preprocessing.preprocess import return_datasplit
from trainers.soft_label_topic_trainer import SoftLabelTrainer


def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def view_stats(annotations):
    # sum hard_label column
    hard_label_sums = np.sum(np.stack(annotations.df["hard_label"].values), axis=0)

    # sum gold column
    gold_sums = np.sum(np.stack(annotations.df["gold"].values), axis=0)

    print("Hard label:")
    print(hard_label_sums)

    print("Gold:")
    print(gold_sums)


def classify_samples(ds, trainer, prepend="test"):
    original_callbacks = trainer.trainer.callback_handler.callbacks
    trainer.trainer.callback_handler.callbacks = [
        callback
        for callback in original_callbacks
        if not isinstance(callback, TrainerCallback)
    ]

    preds = trainer.trainer.predict(ds)
    logits = preds.predictions
    labels = preds.label_ids

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.from_numpy(logits))
    preds = (probs >= 0.5).numpy()

    metrics = {}
    metrics = trainer.report_scores(
        labels, preds, metric_group_prepend=prepend, dict_of_metrics=metrics
    )
    print(f"Metrics for {prepend}")
    print(metrics)

    # return list of np arrays and metrics
    return [pred for pred in preds.astype(int)], metrics


def main():
    # get command line args
    args = parse_args()
    log_args(args)

    # get configs
    config_file = args.config if args.config else "config.yaml"

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # merge configs and args
    config = merge_config(config, args)

    # set random seed
    set_seed(config["seed"])

    # output config to user
    print("Config:")
    print(json.dumps(config, indent=2))

    # create label generator
    topic_label_generator = SemEvalTopicLabelGenerator(config["labels2id"])
    label_cols = list(set([config["train_label_method"], config["test_label_method"]]))

    # load data
    # load train set
    train_df = pd.read_csv(config["train_path"])
    if config["train_setting"] == "chinese_only":
        train_df = train_df[train_df["lang"] == "chinese"]
    elif config["train_setting"] == "sem_only":
        train_df = train_df[train_df["lang"] != "chinese"]
    train_df["sample_weight"] = 1
    train_df = topic_label_generator.binarize_cols(train_df, label_cols)

    train_ds = return_datasplit(
        df=train_df,
        tokenizer_name=config["model_name"],
        label_type=config["train_label_method"],
        text_col=config["text_col"],
        df_only=False,
    )

    # load validation set
    if config["dev_path"]:
        val_df = pd.read_csv(config["dev_path"])
    else:
        val_df = pd.read_csv(config["test_path"])
    val_df["sample_weight"] = 1
    val_df = topic_label_generator.binarize_cols(val_df, label_cols)
    val_ds = return_datasplit(
        df=val_df,
        tokenizer_name=config["model_name"],
        label_type=config["test_label_method"],
        text_col=config["text_col"],
        df_only=False,
    )

    # train
    trainer = SoftLabelTrainer(config, train_ds, val_ds)
    trainer.train()

    full_output_path = (
        f"{config['output_dir_name']}/seed_{config['seed']}/{config['train_setting']}"
    )
    os.makedirs(full_output_path, exist_ok=True)

    # check validation scores
    reval_ds = return_datasplit(
        df=val_df,
        tokenizer_name=config["model_name"],
        label_type=config["test_label_method"],
        text_col=config["text_col"],
        df_only=False,
    )
    val_df["predictions"], metrics = classify_samples(
        reval_ds, trainer, prepend="validation"
    )
    val_df.to_csv(f"{full_output_path}/val_predictions.csv", index=False)

    # prepare test df with all languages
    test_df = pd.read_csv(config["test_path"])
    test_df["sample_weight"] = 1
    test_df = topic_label_generator.binarize_cols(test_df, label_cols)

    # run test for each language
    def run_lang_test(lang):
        # in context to set test_df predictions col
        print(f"Test results for langauge: {lang}")
        lang_mask = test_df["lang"] == lang
        lang_test_df = test_df[lang_mask].copy()
        lang_test_ds = return_datasplit(
            df=lang_test_df,
            tokenizer_name=config["model_name"],
            label_type=config["test_label_method"],
            text_col=config["text_col"],
        )

        predictions, metrics = classify_samples(lang_test_ds, trainer, prepend="test")
        test_df.loc[test_df["lang"] == lang, "predictions"] = pd.Series(
            predictions, index=test_df[lang_mask].index
        )

        return metrics

    langs = list(test_df["lang"].unique())
    print("Langauges to test on...")
    print(langs)
    results_list = []
    for lang in langs:
        metrics = run_lang_test(lang)
        metrics["lang"] = lang
        results_list.append(metrics)
    test_df.to_csv(f"{full_output_path}/test_predictions.csv", index=False)

    with open(f"{full_output_path}/results.txt", "w") as file:
        for res in results_list:
            file.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
