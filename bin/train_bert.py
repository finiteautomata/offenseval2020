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
from sklearn.utils import compute_class_weight
from transformers import (
    AdamW, BertTokenizer, BertModel,
    get_constant_schedule_with_warmup
)

from offenseval.nn import (
    Tokenizer,
    train, evaluate, train_cycle, save_model
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

def create_criterion(train_dataset, device, use_class_weight=True):
    y = [row.subtask_a for row in train_dataset]

    class_weights = compute_class_weight('balanced', ['NOT', 'OFF'], y)

    # normalize it
    class_weights = class_weights / class_weights[0]

    if use_class_weight:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([class_weights[1]]))
    else:
        criterion = nn.BCEWithLogitsLoss()

    criterion = criterion.to(device)
    return criterion

def build_examples(path, fields, mean_threshold):
    """
    Build a list of data.Example from a TSV

    Tries to read accordingly if it has subtask_a, avg, text, tweet...
    """
    df = pd.read_table(path)
    print(df.columns)
    if "id" not in df.columns:
        df["id"] = df[df.columns[0]]
    if "average" in df.columns:
        df["subtask_a"] = "NOT"
        df.loc[df["average"] > mean_threshold, "subtask_a"] = "OFF"
    if "tweet" in df.columns:
        df["text"] = df["tweet"]
    examples = [data.Example.fromdict(t.to_dict(), fields=fields) for _, t in df.iterrows()]
    return examples


def build_dataset(path, fields, mean_threshold):
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


def train_bert(
    model_name, output_path, train_path, dev_path, test_path,
    epochs=5, mean_threshold=0.5):
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

    criterion = create_criterion(train_dataset, device, use_class_weight=True)
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
