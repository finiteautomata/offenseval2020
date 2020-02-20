# import sklearn
# from nltk import tokenize as tokenize_nltk
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import torch.nn as nn
import torch
from sklearn.metrics import precision_recall_fscore_support

def train(model, current_combination, train_loader, val_loader, device, criterion, optimizer, e, losses, accs, language):
    clip=5 # gradient clipping

 
    model.train()
    train_losses = []
 
    running_loss =0 
    running_f = 0

    for inputs, labels in train_loader:

        inputs, labels = inputs.to(device), labels.to(device)

        if current_combination['batch_size'] > len(inputs):
            break

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
    running_f = [0,0]
    running_f_out = 0
    p = 0
    r = 0
    f = 0

    running_precision =  [0,0]
    running_recall =  [0,0]
    for inputs, labels in val_loader:

        inputs, labels = inputs.to(device), labels.to(device)
        if current_combination['batch_size'] > len(inputs):
            break
        output = model(inputs)
        val_loss = criterion(output.squeeze(), labels.float())

        pred = torch.round(output.squeeze()).cpu().detach().numpy()
        target = labels.float().cpu().detach().numpy()   
            
        precision_m, recall_m, fscore_m, support_m = precision_recall_fscore_support(target, pred, average='macro')
        precision, recall, fscore, support = precision_recall_fscore_support(target, pred, average= None)

        p += precision_m
        r += recall_m
        f += fscore_m
        running_loss += val_loss.item()
        
        running_precision[0] += precision[0]
        try:
            running_precision[1] += precision[1]
        except:
            pass

        running_recall[0] += recall[0]
        try:
            running_recall[1] += recall[1]
        except:
            pass

        running_f[0] += fscore[0]
        try:
            running_f[1] += fscore[1]
        except:
            pass

    val_loss = running_loss / len(val_loader)

    return val_loss, running_precision, running_recall, running_f, p, r, f