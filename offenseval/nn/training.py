
import torch
import random
from sklearn.metrics import accuracy_score, f1_score
from tqdm.autonotebook import tqdm
from .evaluation import evaluate

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



def train_cycle(model, optimizer, criterion, scheduler,
                train_it, dev_it, epochs, mean_threshold,
                model_path, early_stopping_tolerance=5):
    best_valid_loss = float('inf')

    max_grad_norm = 1.0

    def get_target(batch):
        return 1. * (batch.avg > mean_threshold)


    pbar = tqdm(range(epochs), ncols=100)
    pbar.set_description("Epochs")

    epochs_without_improvement = 0


    for epoch in pbar:
        print(f"\n\nEpoch {epoch}")
        try:
            train_loss, train_acc = train(
                model, train_it, optimizer, criterion, get_target=get_target,
                max_grad_norm=max_grad_norm, scheduler=scheduler, ncols=100
            )
            valid_loss, valid_acc, valid_f1, pos_f1, neg_f1 = evaluate(
                model, dev_it, criterion, get_target=lambda batch: batch.subtask_a
            )

            desc = f'Train: Loss: {train_loss:.3f} Acc: {train_acc*100:.2f}%'
            desc += f'\nVal. Loss: {valid_loss:.3f} Acc: {valid_acc*100:.2f}% Macro F1 {valid_f1:.3f} (P {pos_f1:.3f} - N {neg_f1:.3f})'

            print(desc)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), model_path)
                print(f"Best model so far (Loss {best_valid_loss:.3f} - Acc {valid_acc:.3f}, F1 {valid_f1:.3f}) saved at {model_path}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_tolerance:
                    print("Early stopping")
                    break
        except KeyboardInterrupt:
            print("Stopping training!")
            break
