import os
import pathlib
import pandas as pd
from tqdm.auto import tqdm
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

def build_examples(path_or_df, fields, mean_threshold=0.15):
    """
    Build a list of data.Example from TSV or dataframe

    Tries to read accordingly if it has subtask_a, avg, text, tweet...

    Arguments:
    ---------

    path_or_df: path or pandas.Dataframe
        If a path, expects a TSV to read examples from it.
        If a pandas.Dataframe, uses it

    fields: dict of column name -> data.Field

    """
    if type(path_or_df) is str:
        path = path_or_df
        df = pd.read_table(path)

    else:
        df = path_or_df

    if "id" not in df.columns:
        df["id"] = df[df.columns[0]]
    if "average" in df.columns:
        df = df[abs(df["average"] - 0.5) > mean_threshold].copy()
        df["subtask_a"] = "NOT"
        df.loc[df["average"] > 0.5, "subtask_a"] = "OFF"
    if "tweet" in df.columns:
        df["text"] = df["tweet"]

    examples = []
    for _, t in tqdm(df.iterrows(), total=len(df)):
        examples.append(data.Example.fromdict(t.to_dict(), fields=fields))

    return examples



def build_dataset(path, fields, mean_threshold=0.5):
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

def build_train_dataset(langs, fields, mean_threshold=0.5):
    """
    Convenient method to build train dataset out of many languages

    langs: list of strings
        List of languages to consume

    fields: list of data.Fields
        Fields to build the datasets with

    mean_threshold: float
        Threshold to use if distant dataset given
    """
    examples = []

    for lang in langs:
        df = pd.read_table(datasets[lang]["train"])
        examples += build_examples(df, fields)

    return data.Dataset(examples, fields.values())
