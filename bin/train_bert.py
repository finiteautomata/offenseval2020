"""
Script to train a BERT model
"""
import os
from datetime import datetime
import fire
import torch
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
    y = [(1*(row.avg > 0.6)) for row in train_dataset]

    class_weights = compute_class_weight('balanced', [0, 1], y)

    # normalize it
    class_weights = class_weights / class_weights[0]

    if use_class_weight:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([class_weights[1]]))
    else:
        criterion = nn.BCEWithLogitsLoss()

    criterion = criterion.to(device)
    return criterion


def train_bert(
    model_name, output_path, train_path, dev_path, test_path,
    epochs=5, mean_threshold=0.6):
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
    AVG = data.LabelField(dtype = torch.float, use_vocab=False, preprocessing=float)
    STD = data.LabelField(dtype = torch.float, use_vocab=False, preprocessing=float)
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



    train_dataset = data.TabularDataset(
        train_path,
        format="tsv", skip_header=True,
        fields=[("id", ID), ("text", TEXT), ("avg", AVG), ("std", STD)],
    )

    dev_dataset = data.TabularDataset(
        dev_path,
        format="tsv", skip_header=True,
        fields=[("id", ID), ("text", TEXT),
                ("subtask_a", SUBTASK_A), ("subtask_b", None), ("subtask_c", None)],

    )

    test_dataset = data.TabularDataset(
        test_path,
        format="tsv", skip_header=True,
        fields=[("id", ID), ("text", TEXT),
                ("subtask_a", SUBTASK_A)],
    )
    SUBTASK_A.build_vocab(dev_dataset)

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

    train_cycle(model, optimizer, criterion, scheduler,
                train_it, dev_it, epochs, mean_threshold=mean_threshold,
                model_path=output_path, early_stopping_tolerance=5)

    print("\n\nLoading best-loss model")
    model.load_state_dict(torch.load(output_path))


    loss, acc, f1, pos_f1, neg_f1 = evaluate(model, dev_it, criterion, get_target=lambda batch: batch.subtask_a)

    print(f'Val Loss: {loss:.3f}  Acc: {acc*100:.2f}% Macro F1: {f1:.3f} Pos F1 {pos_f1:.3f} Neg F1 {neg_f1:.3f}')


    loss, acc, f1, pos_f1, neg_f1 = evaluate(model, test_it, criterion, get_target=lambda batch: batch.subtask_a)

    print(f'Test Loss: {loss:.3f}  Acc: {acc*100:.2f}% Macro F1: {f1:.3f} Pos F1 {pos_f1:.3f} Neg F1 {neg_f1:.3f}')

    save_model(model, TEXT, output_path)


if __name__ == "__main__":
    fire.Fire(train_bert)
