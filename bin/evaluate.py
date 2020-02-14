"""
Script to train a BERT model
"""
import os
from datetime import datetime
import fire
import torch
from torchtext import data
import torch.nn as nn
from transformers import (
    AdamW, BertForSequenceClassification, BertTokenizer,
    get_constant_schedule_with_warmup
)

from offenseval.nn import (
    Tokenizer,
    train, evaluate_dataset, train_cycle, save_model, load_model
)



def evaluate_model(model_path, test_path, batch_size=32):
    """

    Arguments
    ---------
    model_path: a path
        Where the model is. Note that we also expect to find a vocab with a similar name.
        That is, if we have model.pt => we expect to find a vocab at model.vocab.pkl
    test_path: path to test dataset
        Path to test .csv.
    """
    print(f"\nTesting model at {model_path} against {test_path}")
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, TEXT = load_model(model_path, device=device)


    loss, acc, f1, pos_f1, neg_f1 = evaluate_dataset(
        model, TEXT, test_path, batch_size
    )

    print("OBSERVATION: Do not compare this loss with the one of the training! This is not weighted")
    print(f'Test Loss: {loss:.3f}  Acc: {acc*100:.2f}% Macro F1: {f1:.3f} Pos F1 {pos_f1:.3f} Neg F1 {neg_f1:.3f}')

if __name__ == "__main__":
    fire.Fire(evaluate_model)
