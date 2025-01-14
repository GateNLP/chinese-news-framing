import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("data/chinese_news_framing_with_text.csv")

    # separate splits
    train_df = df[df["dataset_splits"] == "train"]
    dev_df = df[df["dataset_splits"] == "dev"]
    test_df = df[df["dataset_splits"] == "test"]

    # write files
    train_df.to_csv("data/chinese_news_framing_train.csv", index=False)
    dev_df.to_csv("data/chinese_news_framing_dev.csv", index=False)
    test_df.to_csv("data/chinese_news_framing_test.csv", index=False)
