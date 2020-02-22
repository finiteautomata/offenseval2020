"""
Script to split dataset into train, dev, test, and a small sample of train
"""
import fire
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def merge_olid():
    """
    Merge OLID test set and labels (for subtask A)
    """
    print("\n\nMerging OLID test set and labels...")

    df_test = pd.read_table("data/olid/testset-levela.tsv", index_col=0)

    labels_a = pd.read_csv(
        "data/olid/labels-levela.csv",
        names=["subtask_a"],
        index_col=0,
        header=None
    )

    df_test = pd.merge(df_test, labels_a, left_index=True, right_index=True)

    assert(all(df_test["subtask_a"].notna()))

    save_path = "data/olid/test_a.tsv"
    df_test = df_test.rename(columns={"tweet": "text"})
    df_test.to_csv(save_path, sep="\t")

    print(f"{df_test.shape[0]} instances saved to {save_path}")


def split_datasets(frac=0.2, random_state=20202020):
    """
    Generate 'dev' splits for datasets.

    Also, merge test labels for OLID dataset
    """

    files = [
        "data/Danish/offenseval-da-training-v1.tsv",
        "data/Greek/offenseval-greek-training-v1.tsv",
        "data/Turkish/offenseval-tr-training-v1.tsv"
    ]

    print("Generating splits")
    print(f"Fraction = {frac:.2f} random_state={random_state}")

    for path in files:
        print(f"\nSplitting {path}")
        df = pd.read_table(path, index_col=0)
        # Remove possibly null lines
        df = df[df["tweet"].notna()]
        dir = os.path.dirname(path)

        print(f"{len(df)} instances read")
        df_train, df_dev = train_test_split(
            df, test_size=frac, stratify=df["subtask_a"],
            random_state=random_state
        )

        train_path = os.path.join(dir, "train.tsv")
        dev_path = os.path.join(dir, "dev.tsv")

        # Save as TSV
        df_train.to_csv(train_path, sep="\t")
        print(f"{df_train.shape[0]} instances saved to {train_path}")

        df_dev.to_csv(dev_path, sep="\t")
        print(f"{df_dev.shape[0]} instances saved to {dev_path}")

    merge_olid()


if __name__ == "__main__":
    fire.Fire(split_datasets)
