import numpy as np
import pandas as pd
from effiara.label_generator import LabelGenerator
from effiara.utils import csv_to_array
from sklearn.preprocessing import MultiLabelBinarizer


class SemEvalTopicLabelGenerator:

    def __init__(self, label_mapping: dict):
        self.label_mapping = label_mapping
        self.num_classes = len(label_mapping)
        self.mlb = MultiLabelBinarizer(classes=list(label_mapping.keys()))
        self.mlb.fit([])  # predefined fit so call with empty list

    def binarize(self, label: list) -> list:
        """Binarise the label.

            Args:
                label (list): list of labels inside label_mapping

        Returns:
                list: list of one-hot labels for each class.
        """
        return self.mlb.transform([label])[0]

    def binarize_cols(self, df: pd.DataFrame, cols_to_binarize: list) -> pd.DataFrame:
        """Binarise all columns with the given label mapping.

        Args:
            df (pd.DataFrame): dataframe containing all data.
            cols_to_binarize (list): list of column names to binarise.

        Returns:
            pd.DataFrame: dataframe with specified columns binarised.

        """
        for col in cols_to_binarize:
            df[col] = df[col].apply(csv_to_array)
            df[col] = df[col].apply(self.binarize)
        return df
