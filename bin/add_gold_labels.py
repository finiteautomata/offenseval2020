"""
Script to split dataset into train, dev, test, and a small sample of train
"""
import fire
import os
import csv
import pandas as pd
from tqdm import tqdm


def set_labels(test_path, labels_path):
    test = pd.read_table(
        test_path, quoting=csv.QUOTE_NONE, index_col=0)
    labels = pd.read_csv(
        labels_path, header=None, index_col=0,
        names=["id", "subtask_a"]
    )

    test["subtask_a"] = labels["subtask_a"]

    print("Checking no row has empty label...")
    assert all(test["subtask_a"].notna())

    test.to_csv(test_path, sep="\t")
    print(f"Test saved to {test_path}\n")


def add_gold_labels():
    """
    Adds gold labels to each test set
    """


    set_labels(
        "data/Arabic/test.tsv",
        "data/gold/arabic-goldlabels.csv",
    )

    set_labels(
        "data/Danish/test.tsv",
        "data/gold/danish-goldlabels.csv",
    )
    set_labels(
        "data/English/test.tsv",
        "data/gold/englishA-goldlabels.csv",
    )

    set_labels(
        "data/Greek/test.tsv",
        "data/gold/greek-goldlabels.csv",
    )

    set_labels(
        "data/Turkish/test.tsv",
        "data/gold/turkish-goldlabels.tsv",
    )
if __name__ == "__main__":
    fire.Fire(add_gold_labels)
