"""
Script to split dataset into train, dev, test, and a small sample of train
"""
import os
from datetime import datetime
import fire
import torch
from torchtext import data
from transformers import BertTokenizer
from offenseval import Tokenizer

def timeit(func):
    def ret_func(*args, **kwargs):
        begin = datetime.now()
        ret = func(*args, **kwargs)
        end = datetime.now()
        delta = end - begin
        print(f"Time {(delta.seconds) // 60}m{delta.seconds % 60}s\n")

        return ret
    return ret_func


@timeit
def print_report(model, dev_it, NUM_SCORE):
    labels, preds = get_labels_and_predictions(model, dev_it)

    y = [float(NUM_SCORE.vocab.itos[t]) for t in labels]
    y_pred = [float(NUM_SCORE.vocab.itos[t]) for t in preds]

    acc, macro_f1, mse, weighted_mse = get_metrics(y, y_pred)

    print(f"Accuracy     = {acc:.3f}")
    print(f"Macro-F1     = {macro_f1:.3f}")
    print(f"MSE          = {mse:.3f}")
    print(f"Weighted MSE = {weighted_mse:.3f}")

def save_model(model, TEXT, NUM_SCORE, output_path):
    base, _ = os.path.splitext(output_path)
    vocab_path = f"{base}.vocab.pkl"
    num_score_path = f"{base}.num_score.pkl"

    torch.save(model, output_path)

    with open(vocab_path, "wb") as f:
        pickle.dump(TEXT, f)

    with open(num_score_path, "wb") as f:
        pickle.dump(NUM_SCORE, f)

    print(f"Model saved to {output_path}")
    print(f"Vocab saved to {vocab_path}")
    print(f"Label vocab saved to {num_score_path}")



def train_bert(output_path, train_path, dev_path, test_path,
    epochs=5):
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

    fix_length: int (default None)
        Truncate the sentences to given length. If None, no cut is performed
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
    return

if __name__ == "__main__":
    fire.Fire(train_bert)
