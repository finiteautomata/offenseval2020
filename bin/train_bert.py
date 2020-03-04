"""
Script to train a BERT model
"""
import os
from datetime import datetime
import fire
import torch
import pandas as pd
from torchtext import data
import torch.optim as optim
import torch.nn as nn
from transformers import (
    AdamW, BertTokenizer, BertModel,
    get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
)
from offenseval.datasets import datasets, build_dataset, build_train_dataset, build_datasets
from offenseval.nn import (
    Tokenizer,
    train, evaluate, train_cycle, save_model, create_criterion,
    create_bert_fields
)
from offenseval.nn.models import BertSeqModel


AVAILABLE_MODELS = {"bert_uncased", "bert_cased"}


def create_model_and_tokenizer(model_name, device):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"{model_name} not available -- must be in {AVAILABLE_MODELS}")

    if model_name == "bert_uncased":
        bert_name = "bert-base-multilingual-uncased"
    elif model_name == "bert_cased":
        bert_name = "bert-base-multilingual-cased"
    else:
        raise ValueError("Must set BERT type")

    print(f"Using {bert_name}")
    bert_model = BertModel.from_pretrained(bert_name)
    bert_tokenizer = BertTokenizer.from_pretrained(bert_name)


    model = BertSeqModel(bert_model)
    model = model.to(device)

    return model, bert_tokenizer



def train_bert(
    model_name, output_path, train_path=None, dev_path=None, test_path=None,
    lang=None, epochs=5, mean_threshold=0.15, use_class_weight=True, batch_size=32,
    monitor="f1", schedule="linear", lr=1):
    """
    Train and save an RNN classifier
    Arguments
    ---------
    output_path: a path
        Where to save the model

    train_path: path to dataset
        Path to train .csv

    dev_path: path to dataset
        Path to dev .csv.

    model_name: string
        Must be "bert_cased", "bert_uncased"

    use_class_weight: bool (default True)
        Use class weight in loss

    monitor: "f1" or "loss"
        Whether to monitor f1 or loss in early stopping regime

    schedule: "linear" or "constant"
        Linear or Constant Scheduler with Warmup

    lr: float
        Will be multiplied by 10^-5
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\nCreating model...")
    model, bert_tokenizer = create_model_and_tokenizer(model_name, device)

    print("Reading and tokenizing data...")


    ID, SUBTASK_A, TEXT = create_bert_fields(bert_tokenizer)


    fields = {
        "id": ('id', ID),
        "text": ('text', TEXT),
        "subtask_a": ("subtask_a", SUBTASK_A)
    }

    #print(f"\n\nTraining BERT using {train_path}. Testing against {dev_path}")
    #print(f"Using mean threshold = {mean_threshold:.2f}")

    train_dataset, dev_dataset, test_dataset = build_datasets(
        lang=lang, train_path=train_path, dev_path=dev_path,
        test_path=test_path, fields=fields, mean_threshold=mean_threshold,
    )

    # For time being, ignore test dataset
    test_dataset = dev_dataset

    SUBTASK_A.build_vocab(dev_dataset)
    print(SUBTASK_A.vocab.itos)
    assert SUBTASK_A.vocab.itos == ["NOT", "OFF"]

    print("Building iterators")

    BATCH_SIZE = batch_size

    train_it, dev_it, test_it = data.BucketIterator.splits(
        (train_dataset, dev_dataset, test_dataset), batch_size=BATCH_SIZE, device=device,
        sort_key = lambda x: len(x.text), sort_within_batch = True,
    )

    print("Creating optimizer, loss and scheduler")

    if use_class_weight:
        print("Using BCE with class weight")
        criterion = create_criterion(device, weight_with=train_dataset)
    else:
        print("Using BCE -- no class weight")
        criterion = create_criterion(device)

    print(f"Learning rate = {lr}e^-5")
    optimizer = AdamW(model.parameters(), lr=lr * 1e-5)

    num_training_steps = epochs * len(train_it)
    num_warmup_steps = num_training_steps // 10
    warmup_proportion = float(num_warmup_steps) / float(num_training_steps)  # 0.1

    print(f"Monitoring {monitor}")

    if schedule == "linear":
        print("Using linear scheduler with warmup")
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif schedule == "constant":
        print("Using constant scheduler with warmup")
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps,
        )
    else:
        raise ValueError("schedule must be 'linear' or 'constant'")

    def get_target(batch):
        return batch.subtask_a.double()

    train_cycle(
        model, optimizer, criterion, scheduler,
        train_it, dev_it, epochs, get_target=get_target,
        model_path=output_path, early_stopping_tolerance=5,
        monitor=monitor,
    )

    print("\n\nLoading best-loss model")
    model.load_state_dict(torch.load(output_path))


    report = evaluate(model, dev_it, criterion, get_target=lambda batch: batch.subtask_a)

    print(f'Val {report}')


    test_report = evaluate(model, test_it, criterion, get_target=lambda batch: batch.subtask_a)

    print(f'Test {report}')

    save_model(model, TEXT, output_path)


if __name__ == "__main__":
    fire.Fire(train_bert)
