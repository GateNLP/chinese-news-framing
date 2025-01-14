import argparse
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from utils import np_string_to_array


def co_occurrence_errors(true, pred, class_names, lower="diff"):
    """Calculate co-occurrence errors. A negative score indicates that
       the model has underpredicted the number of co-occurrences and
       a positive score indicates the model overpredicted the number
       of co-occurrences. As a co-occurrence matrix produces symmetric
       matrix, use lower="true" or lower="pred" for a more informative
       graphic.

    Args:
        true (np.ndarray): stacked numpy array of true labels in the
            shape (num_samples, num_classes).
        pred (np.ndarray): stacked numpy array of predicted labels in
            the shape (num_samples, num_classes).
        class_names (list[str]): list of class_names.
        lower (str): choose either "diff", "true", "pred" to display in
            the lower triangle of the matrix.

    Returns:
        pd.DataFrame: the co-occurrence matrix as a dataframe with the index
            and column headings set as the class names.
    """
    num_classes = true.shape[1]
    if num_classes != len(class_names):
        raise ValueError(
            "Number of classes in true, pred, and class_names must be consistent."
        )

    co_occurrence_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for i, j in combinations(range(num_classes), 2):
        true_co_occurrence = (true[:, i] & true[:, j]).sum()
        pred_co_occurrence = (pred[:, i] & pred[:, j]).sum()

        co_occurrence_matrix[i, j] = pred_co_occurrence - true_co_occurrence

        if lower == "diff":
            co_occurrence_matrix[j, i] = pred_co_occurrence - true_co_occurrence
        elif lower == "true":
            co_occurrence_matrix[j, i] = true_co_occurrence
        elif lower == "pred":
            co_occurrence_matrix[j, i] = pred_co_occurrence
        else:
            raise ValueError(f"lower option {lower} not recognised.")

    return pd.DataFrame(co_occurrence_matrix, index=class_names, columns=class_names)


def get_true_preds(df, true_label_col, preds_col, lang=None):
    """Get the true labels and preds from a dataframe, given
       the columns containing each.

    Args:
        df (pd.DataFrame): dataframe containing true labels and predictions as
            numpy array strings (default save value using np.ndarray and a
            pd.DataFrame).

        true_label_col (str): dataframe column containing the true label.
        preds_col (str): dataframe column containing the prediction.
        lang (Optional[str]): optional string to get the report of an individual
            language.

    Returns:
        (np.ndarray, np.ndarray): tuple of np arrays (true, preds).
    """
    if lang:
        df = df[df["lang"] == lang]
    true = np.stack(df[true_label_col].apply(np_string_to_array).to_numpy())
    predicted = np.stack(df[preds_col].apply(np_string_to_array).to_numpy())
    return true, predicted


def create_co_occurrence_matrix(
    df,
    true_label_col="frames",
    preds_col="predictions",
    lower="diff",
    lang=None,
):
    """Create co-occurrence matrix given the dataframe, true label column, and
       the column containing the predictions. Shows how often certain news frames
       co-occur with others.

    Args:
        df (pd.DataFrame): dataframe containing true labels and predictions as
            numpy array strings (default save value using np.ndarray and a
            pd.DataFrame).

        true_label_col (str): dataframe column containing the true label.
        preds_col (str): dataframe column containing the prediction.
        lang (Optional[str]): optional string to get the report of an individual
            language.

    Returns:
        pd.DataFrame: the co-occurrence matrix as a dataframe with the index
            and column headings set as the class names.
    """
    # create two numpy arrays (true, preds)
    true, predicted = get_true_preds(df, true_label_col, preds_col, lang=lang)

    # define class names
    class_names = [str(i + 1) for i in range(14)]

    # calc co-occurrence errors
    return co_occurrence_errors(true, predicted, class_names, lower=lower)


def plot_co_occurrence_matrix(
    co_occurrence_matrix, class_names, save_path=None, show=True
):
    """Plot the co-occurrence matrix.

    Args:
        co_occurrence_matrix (pd.DataFrame): the co-occurrence matrix as a dataframe.
        class_names (list[str]): list of class names.
        save_path (Optional[str]): leave as None to not save or add file path to save to.
        show (bool): whether to show the plot.
    """
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        co_occurrence_matrix,
        annot=True,
        fmt="d",
        cmap="coolwarm",
        xticklabels=class_names,
        yticklabels=class_names,
        center=0,
    )
    plt.title("Co-occurrence Errors")

    if save_path:
        print(f"Saving co-occurrence matrix to {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse the errors within the topic predictions."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the predictions made by the model.",
    )

    # co-occurrence matrix args
    parser.add_argument(
        "--co_occurrence_matrix",
        action="store_true",
        help="Whether to create a co-occurrence matrix.",
    )
    parser.add_argument(
        "--lower",
        type=str,
        default="diff",
        help="How to display the lower triangle of the co-occurrence matrix.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Where to save the co-occurrence matrix if it is created.",
    )
    parser.add_argument(
        "--noshow",
        action="store_true",
        help="Whether to skip showing the co-occurrence matrix.",
    )

    # classification summary args
    parser.add_argument(
        "--classification_summary",
        action="store_true",
        help="Whether to show the classification summary.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Which language to assess in the classification report.",
    )

    return parser.parse_args()


def co_occurrence_error_analysis(df, lower, show, save_path, lang=None):
    # get co-occurrence matrix
    com = create_co_occurrence_matrix(df, lower=lower, lang=lang)

    # plot co-occurrence matrix
    class_names = [str(i + 1) for i in range(14)]
    plot_co_occurrence_matrix(com, class_names, show=show, save_path=save_path)


def get_classification_report(df, lang=None):
    true, preds = get_true_preds(df, "frames", "predictions", lang=lang)
    return classification_report(true, preds, zero_division=0.0)


def main():
    # args needed
    args = parse_args()

    # load dataframe
    df = pd.read_csv(args.data_path)

    # co-occurrence matrix
    if args.co_occurrence_matrix:
        co_occurrence_error_analysis(
            df, args.lower, (not args.noshow), args.save_path, lang=args.lang
        )

    # classification report
    if args.classification_summary:
        print(get_classification_report(df, lang=args.lang))


if __name__ == "__main__":
    main()
