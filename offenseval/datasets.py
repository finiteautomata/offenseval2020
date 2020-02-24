import os
import pathlib
import pandas as pd
from torchtext import data

# Gets the
_base_dir = pathlib.Path(__file__).parent.absolute().parent
_data_dir = os.path.join(_base_dir, "data")

datasets = {
    "english": {
        "train": os.path.join(_data_dir, "English", "task_a_distant.tsv"),
        "dev": os.path.join(_data_dir, "olid", "olid-training-v1.0.tsv"),
        "test":  os.path.join(_data_dir, "olid", "test_a.tsv"),
    },
    "olid": {
        "train": os.path.join(_data_dir, "olid", "olid-training-v1.0.tsv"),
        "dev": os.path.join(_data_dir, "olid", "test_a.tsv"),
        "test":  os.path.join(_data_dir, "olid", "test_a.tsv"),
    },
    "danish": {
        "train": os.path.join(_data_dir, "Danish", "train.tsv"),
        "dev": os.path.join(_data_dir, "Danish", "dev.tsv"),
        "test":  os.path.join(_data_dir, "Danish", "dev.tsv"),
    },
    "greek": {
        "train": os.path.join(_data_dir, "Greek", "train.tsv"),
        "dev": os.path.join(_data_dir, "Greek", "dev.tsv"),
        "test":  os.path.join(_data_dir, "Greek", "dev.tsv"),
    },
    "arabic": {
        "train": os.path.join(_data_dir, "Arabic", "offenseval-ar-training-v1.tsv"),
        "dev": os.path.join(_data_dir, "Arabic", "offenseval-ar-dev-v1.tsv"),
        "test":  os.path.join(_data_dir, "Arabic", "offenseval-ar-dev-v1.tsv"),
    },
    "turkish": {
        "train": os.path.join(_data_dir, "Turkish", "train.tsv"),
        "dev": os.path.join(_data_dir, "Turkish", "dev.tsv"),
        "test":  os.path.join(_data_dir, "Turkish", "dev.tsv"),
    },
}

def build_examples(path, fields, mean_threshold):
    """
    Build a list of data.Example from a TSV

    Tries to read accordingly if it has subtask_a, avg, text, tweet...
    """
    df = pd.read_table(path)

    if "id" not in df.columns:
        df["id"] = df[df.columns[0]]
    if "average" in df.columns:
        df["subtask_a"] = "NOT"
        df.loc[df["average"] > mean_threshold, "subtask_a"] = "OFF"
    if "tweet" in df.columns:
        df["text"] = df["tweet"]

    examples = [data.Example.fromdict(t.to_dict(), fields=fields) for _, t in df.iterrows()]
    return examples


def build_dataset(path, fields, mean_threshold):
    """
    Builds a dataset from a TSV

    Arguments:
    ----------

    path: a path
        Path to csv

    fields: dict of fields

        Dictionary of the form:

        {
            "column_name_1": ["field_name_1", <data.Field>],
            "column_name_2": ["field_name_1", <data.Field>],
            ...
        }

    mean_threshold:
        If a distant dataset is used, the threshold for offensiveness

    """
    examples = build_examples(path, fields, mean_threshold)
    # Note: It is very strange how the fields are passed to these functions
    return data.Dataset(examples, fields.values())
