"""Module to find tricky samples for certain models."""

import argparse

import numpy as np
import pandas as pd
from utils import list_string_to_array, np_string_to_array

LABEL_MAPPING = {
    "Economic": 1,
    "Capacity_and_resources": 2,
    "Morality": 3,
    "Fairness_and_equality": 4,
    "Legality_Constitutionality_and_jurisprudence": 5,
    "Policy_prescription_and_evaluation": 6,
    "Crime_and_punishment": 7,
    "Security_and_defense": 8,
    "Health_and_safety": 9,
    "Quality_of_life": 10,
    "Cultural_identity": 11,
    "Public_opinion": 12,
    "Political": 13,
    "External_regulation_and_reputation": 14,
}


def topics_from_binary(
    bin_array: np.ndarray, label_mapping: dict = LABEL_MAPPING
) -> str:
    """Take a one-hot binary representation of the topics
       and produce a list of the topics based on a label
       mapping.

    Args:
        bin_array (np.ndarray): array containing 1s and 0s
            for whether a given topic is present in the text
            e.g. [1 0 0 1 0 1 0 1 0 0 0 1 0 1]
        label_mapping (dict): mapping of topic names to indices.

    Returns:
        str: string of words from the dict (based on the index
            of topics with a 1 in the binary representation).
    """
    # convert to index lookup
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    topics = [
        reverse_label_mapping[i + 1]
        for i, classification in enumerate(bin_array)
        if classification == 1
    ]
    return ", ".join(topics)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse the errors within the topic predictions."
    )

    parser.add_argument(
        "--model_1_name", type=str, required=True, help="Name of model 1."
    )
    parser.add_argument(
        "--model_2_name", type=str, required=True, help="Name of model 2."
    )
    parser.add_argument(
        "--model_1_predictions",
        type=str,
        required=True,
        help="Path to the predictions made by model 1.",
    )
    parser.add_argument(
        "--model_2_predictions",
        type=str,
        required=True,
        help="Path to the predictions made by model 2.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Where to save the tricky samples to (full path).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    model_1 = args.model_1_name
    model_2 = args.model_2_name

    model_1_df_path = args.model_1_predictions
    model_2_df_path = args.model_2_predictions

    # load two dataframes
    model_1_df = pd.read_csv(model_1_df_path)
    model_2_df = pd.read_csv(model_2_df_path)

    # rename predictions in first to {model_1}_predictions
    model_1_df.rename(columns={"predictions": f"{model_1}_predictions"}, inplace=True)

    # add second predictions as new column in first as {model_2}_predictions
    model_1_df[f"{model_2}_predictions"] = model_2_df["predictions"]

    # convert numpy strings to arrays

    # NOTE: change to list_string_to_array if errors due to structure
    model_1_df["frames"] = model_1_df["frames"].apply(np_string_to_array)
    model_1_df[f"{model_1}_predictions"] = model_1_df[f"{model_1}_predictions"].apply(
        np_string_to_array
    )
    model_1_df[f"{model_2}_predictions"] = model_1_df[f"{model_2}_predictions"].apply(
        np_string_to_array
    )

    # correctness scores for both models (percentage correctness)
    model_1_df[f"{model_1}_correctness"] = model_1_df.apply(
        lambda row: np.mean(row[f"{model_1}_predictions"] == row["frames"]), axis=1
    )
    model_1_df[f"{model_2}_correctness"] = model_1_df.apply(
        lambda row: np.mean(row[f"{model_2}_predictions"] == row["frames"]), axis=1
    )

    # create a new dataframe to store tricky samples
    tricky_samples = []
    for _, row in model_1_df.iterrows():
        model_1_correctness = row[f"{model_1}_correctness"]
        model_2_correctness = row[f"{model_2}_correctness"]

        # if significant difference add to tricky samples
        correctness_diff = model_1_correctness - model_2_correctness
        # want model_1_correctness close to 1 and model_2_correctness closer to 0
        if correctness_diff > 0.2:  # threshold can be adapted
            tricky_samples.append(row)

    tricky_samples_df = pd.DataFrame(tricky_samples)

    tricky_samples_df["gold_standard"] = tricky_samples_df["frames"].apply(
        topics_from_binary
    )
    tricky_samples_df[f"{model_1}_preds"] = tricky_samples_df[
        f"{model_1}_predictions"
    ].apply(topics_from_binary)
    tricky_samples_df[f"{model_2}_preds"] = tricky_samples_df[
        f"{model_2}_predictions"
    ].apply(topics_from_binary)

    cols_to_keep = [
        "text",
        "clean_text",
        "gold_standard",
        f"{model_1}_preds",
        f"{model_2}_preds",
    ]

    tricky_samples_df = tricky_samples_df[cols_to_keep]

    print(tricky_samples_df)

    # save tricky samples to csv
    tricky_samples_df.to_csv(args.save_path, index=False)
    print(f"Tricky samples saved to {args.save_path}.")


if __name__ == "__main__":
    main()
