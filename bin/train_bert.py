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
    get_constant_schedule_with_warmup
)
from offenseval.datasets import datasets, build_dataset
from offenseval.nn import (
    Tokenizer,
    train, evaluate, train_cycle, save_model, create_criterion
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

def get_paths(lang, train_path, dev_path, test_path):
    if bool(lang) == bool(train_path and test_path and dev_path):
        raise ValueError("You must define either --lang or --train_path, --dev_path, --test_path but not both")

    if lang:
        if lang not in datasets:
            raise ValueError(f"lang must be one of {datasets.keys()}")

        lang_datasets = datasets[lang]
        return lang_datasets["train"], lang_datasets["dev"], lang_datasets["train"]
    else:
        return train_path, dev_path, test_path


def train_bert(
    model_name, output_path, train_path=None, dev_path=None, test_path=None,
    lang=None, epochs=5, mean_threshold=0.5):
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
    """

    train_path, dev_path, test_path = get_paths(
        lang, train_path, dev_path, test_path
    )
    print(f"\n\nTraining BERT using {train_path}. Testing against {dev_path}")
    print(f"Using mean threshold = {mean_threshold:.2f}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\nCreating model...")
    model, bert_tokenizer = create_model_and_tokenizer(model_name, device)

    print("Reading and tokenizing data...")

    init_token_idx = bert_tokenizer.cls_token_id
    eos_token_idx = bert_tokenizer.sep_token_id
    pad_token_idx = bert_tokenizer.pad_token_id
    unk_token_idx = bert_tokenizer.unk_token_id

    # Trying to cut this down to check if this improves memory usage

    tokenizer = Tokenizer(bert_tokenizer)

    ID = data.Field(sequential=False, use_vocab=False)
    # All these arguments are because these are really floats
    # See https://github.com/pytorch/text/issues/78#issuecomment-541203609
    SUBTASK_A = data.LabelField()

    TEXT = data.Field(
        tokenize=tokenizer.tokenize,
        include_lengths = True,
        use_vocab=False,
        batch_first = True,
        preprocessing = tokenizer.convert_tokens_to_ids,
        init_token = init_token_idx,
        eos_token = eos_token_idx,
        pad_token = pad_token_idx,
        unk_token = unk_token_idx
    )


    fields = {
        "id": ('id', ID),
        "text": ('text', TEXT),
        "subtask_a": ("subtask_a", SUBTASK_A)
    }


    train_dataset = build_dataset(train_path, fields, mean_threshold)
    dev_dataset = build_dataset(dev_path, fields, mean_threshold)
    test_dataset = build_dataset(test_path, fields, mean_threshold)

    SUBTASK_A.build_vocab(dev_dataset)
    print(SUBTASK_A.vocab.itos)
    assert SUBTASK_A.vocab.itos == ["NOT", "OFF"]

    print("Building iterators")

    BATCH_SIZE = 32

    train_it, dev_it, test_it = data.BucketIterator.splits(
        (train_dataset, dev_dataset, test_dataset), batch_size=BATCH_SIZE, device=device,
        sort_key = lambda x: len(x.text), sort_within_batch = True,
    )

    print("Creating optimizer, loss and scheduler")

    criterion = create_criterion(device, weight_with=train_dataset)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    num_training_steps = epochs * len(train_it)
    num_warmup_steps = num_training_steps // 10
    warmup_proportion = float(num_warmup_steps) / float(num_training_steps)  # 0.1

    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
    )

    def get_target(batch):
        return batch.subtask_a.double()

    train_cycle(
        model, optimizer, criterion, scheduler,
        train_it, dev_it, epochs, get_target=get_target,
        model_path=output_path, early_stopping_tolerance=5,
    )

    print("\n\nLoading best-loss model")
    model.load_state_dict(torch.load(output_path))


    loss, acc, f1, pos_f1, neg_f1 = evaluate(model, dev_it, criterion, get_target=lambda batch: batch.subtask_a)

    print(f'Val Loss: {loss:.3f}  Acc: {acc*100:.2f}% Macro F1: {f1:.3f} Pos F1 {pos_f1:.3f} Neg F1 {neg_f1:.3f}')


    loss, acc, f1, pos_f1, neg_f1 = evaluate(model, test_it, criterion, get_target=lambda batch: batch.subtask_a)

    print(f'Test Loss: {loss:.3f}  Acc: {acc*100:.2f}% Macro F1: {f1:.3f} Pos F1 {pos_f1:.3f} Neg F1 {neg_f1:.3f}')

    save_model(model, TEXT, output_path)


if __name__ == "__main__":
    fire.Fire(train_bert)
