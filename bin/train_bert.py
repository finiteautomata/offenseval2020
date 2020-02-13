"""
Script to split dataset into train, dev, test, and a small sample of train
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
    AdamW, BertForSequenceClassification, BertTokenizer,
    get_constant_schedule_with_warmup
)

from offenseval.nn import (
    Tokenizer,
    train, evaluate, train_cycle, save_model
)


def create_model(device):
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-multilingual-uncased',
        num_labels=1,
    )

    model = model.to(device)

    return model

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
    output_path, train_path, dev_path, test_path,
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
    """
    print(f"\n\nTraining BERT using {train_path}. Testing against {dev_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Reading and tokenizing data...")

    begin = datetime.now()
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
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


    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_it, dev_it, test_it = data.BucketIterator.splits(
        (train_dataset, dev_dataset, test_dataset), batch_size=BATCH_SIZE, device=device,
        sort_key = lambda x: len(x.text), sort_within_batch = True,
    )

    print("Creating model, optimizer, loss and scheduler")

    model = create_model(device)
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
