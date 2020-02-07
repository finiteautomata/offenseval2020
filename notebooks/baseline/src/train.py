# import sklearn
# from nltk import tokenize as tokenize_nltk
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import torch.nn as nn
import torch
def train(model, current_combination, train_loader, val_loader, device, criterion, optimizer, e, losses, accs):
    clip=5 # gradient clipping

 
    model.train()
    train_losses = []
 
    running_loss =0 
    running_f = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if current_combination['batch_size'] > len(inputs):
            break
        # print(current_combination['batch_size'])
        model.zero_grad()
        output = model(inputs)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        pred = torch.round(output.squeeze()).cpu().detach().numpy()
        target = labels.float().cpu().detach().numpy()   
        running_loss += loss.item()
        running_f += f1_score(target,pred)

    train_loss = running_loss / len( train_loader)
    train_f = running_f / len( train_loader)
    
    losses['train'].append(train_loss)
    accs['train'].append(train_f)

    val_losses = []
    model.eval()
    y_pred = []
    y_true = []
    running_loss = 0
    running_f = 0

    for inputs, labels in val_loader:

        inputs, labels = inputs.to(device), labels.to(device)
        if current_combination['batch_size'] > len(inputs):
            break
        output = model(inputs)
        val_loss = criterion(output.squeeze(), labels.float())

        pred = torch.round(output.squeeze()).cpu().detach().numpy()
        target = labels.float().cpu().detach().numpy()   

        running_f += f1_score(target,pred)
        running_loss += val_loss.item()

    val_loss = running_loss / len(val_loader)
    val_f = running_f/ len(val_loader)
    losses['validation'].append(val_loss)
    accs['validation'].append(val_f)
    return val_loss, val_f, losses, accs