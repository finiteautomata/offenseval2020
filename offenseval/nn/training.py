import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm.autonotebook import tqdm

def train(model, iterator, optimizer, criterion, get_target,
          scheduler=None, max_grad_norm=None, ncols=500):
    """
    Trains the model for one full epoch

    Arguments:

    model: torch.nn.Module
        Model to be trained

    iterator:
        An iterator over the train batches

    optimizer: torch.nn.optimizer
        An optimizer

    criterion:
        Loss function

    scheduler: (optional) A scheduler
        Scheduler that will be called (if given) after each call to `optimizer.step()`

    get_target: a function
        Function receiving a batch and returning the targets

    max_grad_norm: float (optional, default None)
        If not none, applies gradient clipping using the given norm

    """
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    pbar = tqdm(enumerate(iterator), total=len(iterator), ncols=ncols)
    for i, batch in pbar:
        # Zero gradients first
        optimizer.zero_grad()
        # We assume we always get the length
        text, lens = batch.text
        #target = 1. * (batch.avg > 0.6)
        target = get_target(batch)

        predictions = model(text)
        if type(predictions) is tuple:
            # This is because of BERTSequenceClassifier, sorry!
            predictions = predictions[0]

        loss = criterion(predictions.view(-1), target)
        # Calculate gradients
        loss.backward()
        # Gradient clipping
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        if scheduler:
            scheduler.step()

        # Calculate metrics
        prob_predictions = torch.sigmoid(predictions)
        preds = torch.round(prob_predictions).detach().cpu()
        acc = accuracy_score(preds, target.cpu())

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        # Update Pbar
        lr = optimizer.param_groups[0]["lr"]

        desc = f"Loss {epoch_loss / (i+1):.3f} -- Acc {epoch_acc / (i+1):.3f} -- LR {lr:.5f}"
        pbar.set_description(desc)

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


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
