"""
Script to split dataset into train, dev, test, and a small sample of train
"""
import fire
import os
import json
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm

def save_sample(df, frac, path, random_state):
    sample = df.sample(frac=frac, random_state=random_state)
    sample.to_csv(path, sep="\t")
    print(f"\n{sample.shape[0] / 1e3:.2f}K records saved to {path}")


def generate_samples(sample_frac=0.05, xsmall_frac=0.0001, random_state=20202020):
    """
    Generate samples for datasets
    """


    path = "data/English/task_a_distant.tsv"

    print(f"\nGenerating sample for {path} (random state = {random_state})")
    print("Loading data...")
    df_train = pd.read_table(path, index_col=0, quoting=csv.QUOTE_NONE)

    print(f"\nGenerating sample of frac = {sample_frac}")
    base, ext = os.path.splitext(path)

    sample_path = f"{base}.sample{ext}"
    save_sample(df_train, sample_frac, sample_path, random_state)

    xsmall_path = f"{base}.xsmall{ext}"
    save_sample(df_train, xsmall_frac, xsmall_path, random_state)



if __name__ == "__main__":
    fire.Fire(generate_samples)
