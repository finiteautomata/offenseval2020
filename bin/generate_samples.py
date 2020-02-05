"""
Script to split dataset into train, dev, test, and a small sample of train
"""
import fire
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

def generate_samples(sample_frac=0.05, random_state=20202020):
    """
    Generate samples for datasets
    """


    path = "data/English/task_a_distant.tsv"

    print(f"\nGenerating sample for {path} (random state = {random_state})")
    print("Loading data...")
    df_train = pd.read_table(path, index_col=0)

    print(f"\nGenerating sample of frac = {sample_frac}")
    base, ext = os.path.splitext(path)
    sample_path = f"{base}.sample{ext}"

    sample = df_train.sample(frac=sample_frac, random_state=random_state)
    sample.to_csv(sample_path, sep="\t")

    print(f"\n{sample.shape[0] / 1e3:.2f}K records saved to {sample_path}")



if __name__ == "__main__":
    fire.Fire(generate_samples)
