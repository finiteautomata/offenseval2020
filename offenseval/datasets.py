import os
import csv
import pathlib
import pandas as pd
from tqdm.auto import tqdm
from torchtext import data

# Gets the
_base_dir = pathlib.Path(__file__).parent.absolute().parent
_data_dir = os.path.join(_base_dir, "data")
_trans_dir = os.path.join(_data_dir, "translations")
datasets = {
    "english": {
        "train": os.path.join(_data_dir, "English", "task_a_distant.tsv"),
        "dev": os.path.join(_data_dir, "olid", "test_a.tsv"),
        "test":  os.path.join(_data_dir, "English", "test.tsv"),
    },
    "olid": {
        "train": os.path.join(_data_dir, "olid", "olid-training-v1.0.tsv"),
        "dev": os.path.join(_data_dir, "olid", "test_a.tsv"),
        "test":  os.path.join(_data_dir, "olid", "test_a.tsv"),
    },
    "danish": {
        "train": os.path.join(_data_dir, "Danish", "train.tsv"),
        "dev": os.path.join(_data_dir, "Danish", "dev.tsv"),
        "test":  os.path.join(_data_dir, "Danish", "test.tsv"),
    },
    "greek": {
        "train": os.path.join(_data_dir, "Greek", "train.tsv"),
        "dev": os.path.join(_data_dir, "Greek", "dev.tsv"),
        "test":  os.path.join(_data_dir, "Greek", "test.tsv"),
    },
    "arabic": {
        "train": os.path.join(_data_dir, "Arabic", "offenseval-ar-training-v1.tsv"),
        "dev": os.path.join(_data_dir, "Arabic", "offenseval-ar-dev-v1.tsv"),
        "test":  os.path.join(_data_dir, "Arabic", "test.tsv"),
    },
    "turkish": {
        "train": os.path.join(_data_dir, "Turkish", "train.tsv"),
        "dev": os.path.join(_data_dir, "Turkish", "dev.tsv"),
        "test":  os.path.join(_data_dir, "Turkish", "test.tsv"),
    },


    "danish-trans": {
        "train": os.path.join(_trans_dir, "Danish", "train.tsv"),
    },
    "greek-trans": {
        "train": os.path.join(_trans_dir, "Greek", "train.tsv"),
    },
    "arabic-trans": {
        "train": os.path.join(_trans_dir, "Arabic", "train.tsv"),
    },
    "turkish-trans": {
        "train": os.path.join(_trans_dir, "Turkish", "train.tsv"),
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
        df = pd.read_table(path, quoting=csv.QUOTE_NONE)
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

    langs: list of strings|
        List of languages to consume

    fields: list of data.Fields
        Fields to build the datasets with

    mean_threshold: float
        Threshold to use if distant dataset given
    """
    examples = []

    for lang in langs:
        perc = None

        if "." in lang:
            lang, perc = lang.split(".")

        dataset_path = datasets[lang]["train"]
        print(f"Reading {dataset_path}", end="")
        df = pd.read_table(dataset_path, quoting=csv.QUOTE_NONE)

        if perc:
            num = (int(perc) * len(df)) // 100
            orig_len = len(df)
            df = df.iloc[:num]
            print(f" at {perc}% -- using {len(df)} instances out of {orig_len}")
        else:
            print(f" at 100% -- using {len(df)} instances")

        examples += build_examples(df, fields)

    return data.Dataset(examples, fields.values())

def build_datasets(
    fields, mean_threshold=None,
    lang=None, train_path=None, dev_path=None, test_path=None,
    ):
    if bool(lang) == bool(train_path):
        raise ValueError("You must define either --lang or --train_path")

    ret = []

    if lang:
        if type(lang) is str and lang[0] != "[":
            """
            Single language
            """
            if lang == "all":
                langs = ["olid", "danish", "turkish", "arabic", "greek"]
            elif lang == "all+trans":
                langs = [
                    "olid", "danish", "turkish", "arabic", "greek",
                    "danish-trans", "turkish-trans", "arabic-trans", "greek-trans",
                ]
            else:
                langs = [lang]
        elif type(lang) is str and lang[0] == "[":
            """
            Workaround
            """
            langs = lang[1:-1].split(",")
        else:
            langs = lang


        for lang in langs:
            if "." in lang:
                """
                There is a modificator for
                """
                lang, perc = lang.split(".")

                if not 0 < int(perc) <= 100:
                    raise ValueError(f"percentage of language must be between 1 and 100")
            if lang not in datasets:
                raise ValueError(f"lang must be one of {datasets.keys()} -- was {lang}")

        print(f"Building from langs {' '.join(langs)}")
        ret.append(build_train_dataset(langs, fields, mean_threshold))

        if dev_path:
            print(f"Using dev set {dev_path}")
            ret.append(build_dataset(dev_path, fields, mean_threshold))
        else:
            print(f"Using dev lang {langs[0]}")
            ret.append(
                build_dataset(
                    datasets[langs[0].split(".")[0]]["dev"],
                    fields,
                    mean_threshold
                )
            )

        if test_path:
            print(f"Using dev set {test_path}")
            ret.append(build_dataset(test_path, fields, mean_threshold))
        else:
            print(f"Using test lang {langs[0]}")
            ret.append(
                build_dataset(
                    datasets[langs[0].split(".")[0]]["test"],
                    fields,
                    mean_threshold
                )
            )

    else:
        ret = []
        ret.append(build_dataset(train_path, fields, mean_threshold))
        ret.append(build_dataset(dev_path, fields, mean_threshold))
        if test_path:
            ret.append(build_dataset(test_path, fields, mean_threshold))

    print(f"Training on {len(ret[0]) / 1000:.3f}K instances")
    return tuple(ret)
