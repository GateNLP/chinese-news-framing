import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def np_string_to_array(npstring: str) -> np.ndarray:
    """Convert a numpy array string back to numpy array.

    Args:
        npstring (str): numpy array in string format to be saved
            in a pd.DataFrame; e.g. [1 0 1 0 0 0 1 0 1].

    Returns
        np.ndarray: numpy array of the string representation.
    """
    # remove braces
    clean_string = npstring.strip()[1:-1]

    # create np array
    return np.fromstring(clean_string, dtype=int, sep=" ")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Display the classification results from the set of predictions in a given file."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the predictions made by the model.",
    )
    return parser.parse_args()


def main():
    # process args
    args = parse_args()
    df_path = args.data_path

    # load dataframe
    df = pd.read_csv(df_path)

    # check the data type of each column
    df["predictions"] = df["predictions"].apply(np_string_to_array)
    df["frames"] = df["frames"].apply(np_string_to_array)

    # get the f1 metrics, etc.
    langs = list(df["lang"].unique())
    for lang in langs:
        lang_mask = df["lang"] == lang
        preds = np.stack(df.loc[lang_mask, "predictions"].to_numpy())
        labels = np.stack(df.loc[lang_mask, "frames"].to_numpy())

        f1_micro = f1_score(labels, preds, average="micro", zero_division=0.0)
        print(f"For language {lang}:")
        print(f1_micro)


if __name__ == "__main__":
    main()
