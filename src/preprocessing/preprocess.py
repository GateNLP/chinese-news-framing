"""
Preprocssor for data used to generate soft labels,
tokenise inputs, return specific folds.
"""

from datasets import Dataset
from transformers import AutoTokenizer


def convert_df_to_tokenized_ds(
    df,
    tokenizer_name,
    text_col="text",
    label_col="label",
    weight_col="sample_weight",
):
    """
    Tokenize a pair of claim and text and create a Huggingface Dataset object.

    Parameters:
        df (DataFrame): The Dataframe containing claim, text and soft labels
        tokenizer_name (str): The pre-trained tokenizer to apply
        claim_col (str): the column in the DataFrame containing the claim
        tweet_col (str): the column in the DataFrame containing tweet text

    Returns:
        Dataset: the Dataset prepared for training
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = Dataset.from_pandas(df)

    # tokenise the text and retain the labels
    def tokenize_function(examples):
        tokenized_inputs = tokenizer(
            examples[text_col],
            truncation=True,
            padding="max_length",
            max_length=512,  # should add this as a config option
        )
        # add labels to the tokenised inputs
        tokenized_inputs["labels"] = examples[label_col]
        tokenized_inputs["sample_weight"] = examples[weight_col]

        return tokenized_inputs

    # apply the tokenisation and remove cols
    ds = ds.map(tokenize_function, batched=True)
    columns_to_keep = ["input_ids", "attention_mask", "labels", "sample_weight"]
    columns_to_remove = [col for col in ds.column_names if col not in columns_to_keep]

    # remove unwanted columns
    ds = ds.remove_columns(columns_to_remove)

    print("Tokenization successful.")

    return ds


def return_datasplit(df, tokenizer_name, label_type, text_col="text", df_only=False):
    """Return data split (train, val, test). Requires
       the dataframe to already contain only the samples
       from this datasplit.

    Args:
        df (pd.DataFrame): dataframe containing samples from
            the required data split (train, val, test).
        label_type (str): label type to be generated
            ("soft_label", "hard_label", "gold").
        text_col (str): column containing the text to be classified.
        df_only: whether to return the dataframe rather than dataset.
    Returns:
        Optional[pd.DataFrame, Dataset]: dataframe or dataset depending on options.
    """
    # set the label
    df.loc[:, "label"] = df[label_type]

    # return df if only df needed
    if df_only:
        return df

    # create dataset
    ds = convert_df_to_tokenized_ds(df, tokenizer_name, text_col=text_col)

    return ds
