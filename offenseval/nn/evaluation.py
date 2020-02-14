import random
import torch
import torch.nn as nn
from torchtext import data
from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, iterator, criterion, get_target):
    """
    Evaluates the model on the given iterator
    """
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        predicted_probas = []
        labels = []
        for batch in iterator:
            text, lens = batch.text
            target = get_target(batch)

            predictions = model(text)[0]
            loss = criterion(predictions.squeeze(1), target.float())

            prob_predictions = torch.sigmoid(predictions)

            predicted_probas.append(prob_predictions)
            labels.append(target.cpu())

            epoch_loss += loss.item()

        predicted_probas = torch.cat(predicted_probas).cpu()
        labels = torch.cat(labels).cpu()

        preds = torch.round(predicted_probas)

        pos_f1 = f1_score(labels, preds)
        neg_f1 = f1_score(1-labels, 1-preds)
        avg_f1 = (pos_f1 + neg_f1) / 2
        acc = accuracy_score(labels, preds)

    return epoch_loss / len(iterator), acc, avg_f1, pos_f1, neg_f1


def evaluate_dataset(model, TEXT, test_path, batch_size=32):
    """
    High level function that evaluates a model on a given dataset

    Arguments:
    ---------

    model: Pytorch Module
        An already trained model

    TEXT: torchtext.data.Field
        Field for the "TEXT" field.

    test_path: path to a .csv

        The dataset must contain the following columns
            - id
            - text/tweet
            - subtask_a (label NOT or OFF)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    return evaluate(model, test_it, criterion, get_target=lambda batch: batch.subtask_a)
