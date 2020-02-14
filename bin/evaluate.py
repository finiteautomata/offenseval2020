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
    train, evaluate, train_cycle, save_model, load_model
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading model...")
    model, TEXT = load_model(model_path)

    model.eval()

    print("Loading dataset...")
    ID = data.Field(sequential=False, use_vocab=False)
    SUBTASK_A = data.LabelField()

    test_dataset = data.TabularDataset(
        test_path,
        format="tsv", skip_header=True,
        fields=[("id", ID), ("text", TEXT),
                ("subtask_a", SUBTASK_A)],
    )
    SUBTASK_A.build_vocab(test_dataset)

    assert SUBTASK_A.vocab.itos == ["NOT", "OFF"]

    print("Building iterators")

    test_it = data.BucketIterator(
        test_dataset, batch_size=batch_size, device=device,
        sort_key = lambda x: len(x.text), sort_within_batch = True,
    )

    # OBSERVATION: Do not compare this loss with the one of the training!
    # This has no class weights

    criterion = nn.BCEWithLogitsLoss()
    loss, acc, f1, pos_f1, neg_f1 = evaluate(model, test_it, criterion, get_target=lambda batch: batch.subtask_a)

    print("OBSERVATION: Do not compare this loss with the one of the training! This is not weighted")
    print(f'Test Loss: {loss:.3f}  Acc: {acc*100:.2f}% Macro F1: {f1:.3f} Pos F1 {pos_f1:.3f} Neg F1 {neg_f1:.3f}')

if __name__ == "__main__":
    fire.Fire(evaluate_model)
