"""
Script to train a BERT model
"""
import os
import subprocess
import csv
import fire
from datetime import datetime
import torch
import pandas as pd
from tqdm.auto import tqdm
from torchtext import data
from offenseval.nn import (
    Tokenizer,
    train, evaluate_dataset, train_cycle, save_model, load_model
)
from offenseval.datasets import build_dataset, datasets

def get_num_lines(test_path):
    with open(test_path, "r") as f:
         return len([1 for l in f])

def get_test_and_output_path(model_path, test_path, output_path, lang):
    if not test_path:
        if not lang:
            raise ValueError("lang must be provided if no test_path given")
        try:
            test_path = datasets[lang]["test"]
        except KeyError:
            print(f"lang must be in {datasets.keys()}")

    if not output_path:
        if not lang or lang not in datasets.keys():
            raise ValueError("lang must be provided if not output_path -- and must be valid language")


        model_name = os.path.basename(model_path)
        without_ext = os.path.splitext(model_name)[0]
        output_path = os.path.join(
            "submissions",
            lang.capitalize(),
            f"{without_ext}.{lang}.csv"
        )

    return test_path, output_path


def generate_submission(model_path, test_path=None, output_path=None, lang=None, batch_size=1):
    """
    Generate submission from model and test file
    Arguments
    ---------
    model_path: a path
        Where the model is. Note that we also expect to find a vocab with a similar name.
        That is, if we have model.pt => we expect to find a vocab at model.vocab.pkl

    test_path: path to test dataset
        Path to test .csv.

    output_path: path to output
    """
    test_path, output_path = get_test_and_output_path(model_path, test_path, output_path, lang)
    print(f"\nGenerating submission for model at {model_path} against {test_path}")
    print(f"Saving at {output_path}")
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, TEXT = load_model(model_path, device=device)

    fields = {
        "text": ('text', TEXT),
    }

    test_dataset = build_dataset(test_path, fields)

    print("Building iterators")

    test_it = data.Iterator(
        test_dataset, batch_size=1, device=device,
        shuffle=False,
    )

    predicted_labels = []
    model.eval()
    print("\nPredicting labels...")
    with torch.no_grad():
        for batch in tqdm(test_it):
            text, _ = batch.text

            predictions = model(text)
            predictions = torch.round(torch.sigmoid(predictions))

            predicted_labels.append(predictions)


    predicted_labels = torch.cat(predicted_labels).cpu().numpy()
    #
    df_test = pd.read_table(test_path, index_col=0, quoting=csv.QUOTE_NONE)

    # Checking number of lines
    num_lines = get_num_lines(test_path)
    if (num_lines-1) != df_test.shape[0] or (num_lines-1) != predicted_labels.shape[0]:
        raise ValueError(
            f"Mismatch in lines: {num_lines-1} in table, {df_test.shape[0]} read from pandas, {predicted_labels.shape[0]} predictions"
        )

    df_test["pred"] = 'NOT'
    df_test.loc[predicted_labels.reshape(-1) > 0, "pred"] = "OFF"

    out = df_test["pred"]
    out.to_csv(output_path, header=False)

    os.system(f"zip -j -r {output_path}.zip {output_path}")

    print(f"{len(df_test)} saved to {output_path}")

if __name__ == "__main__":
    fire.Fire(generate_submission)
