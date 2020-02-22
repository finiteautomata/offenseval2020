import os
import pathlib

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
