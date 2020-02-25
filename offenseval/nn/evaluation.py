import random
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torchtext import data
from sklearn.metrics import accuracy_score, f1_score

def get_outputs(model, iterator, criterion=None, get_target=None):
    """
    Returns the loss and outputs of the model for the given iterator


    Returns
    -------
    predicted_probas: torch.tensor
        Predicted probabilities

    labels: torch.tensor
        True labels
    """
    if not get_target:
        get_target = lambda b: b.subtask_a.float()

    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        predicted_probas = []
        labels = []
        for batch in iterator:
            text, lens = batch.text
            target = get_target(batch)

            predictions = model(text)

            if criterion:
                loss = criterion(predictions.squeeze(1), target.float())
                epoch_loss += loss.item()

            prob_predictions = torch.sigmoid(predictions)
            predicted_probas.append(prob_predictions)
            labels.append(target.cpu())

    predicted_probas = torch.cat(predicted_probas).cpu()
    labels = torch.cat(labels).cpu()

    if criterion:
        return predicted_probas, labels, epoch_loss / len(iterator)
    else:
        return predicted_probas, labels

def evaluate(model, iterator, criterion, get_target):
    """
    Evaluates the model on the given iterator
    """

    predicted_probas, labels, loss = get_outputs(
        model, iterator, criterion, get_target
    )

    preds = torch.round(predicted_probas)

    pos_f1 = f1_score(labels, preds)
    neg_f1 = f1_score(1-labels, 1-preds)
    avg_f1 = (pos_f1 + neg_f1) / 2
    acc = accuracy_score(labels, preds)

    return loss, acc, avg_f1, pos_f1, neg_f1


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

    test_it = data.Iterator(
        test_dataset, batch_size=batch_size, device=device,
        shuffle=False,
    )

    # OBSERVATION: Do not compare this loss with the one of the training!
    # This has no class weights

    criterion = nn.BCEWithLogitsLoss()

    return evaluate(model, tqdm(test_it), criterion, get_target=lambda batch: batch.subtask_a)


def predict_sentence(model, TEXT, sentence):
    bert_tokenizer = TEXT.tokenize.__self__.bert_tokenizer
    # a bit hacky...
    device = next(model.parameters()).device

    model.eval()
    inp = torch.tensor(bert_tokenizer.encode(sentence)).view(1, -1).to(device)

    return torch.sigmoid(model(inp)).item()
