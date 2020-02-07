import sys
# sys.path.append('/gdrive/My Drive/Workspace/Code/Deep Learning Hate Speech/models')
sys.path.append('/gdrive/My Drive/offenseval2020-master/code/src')

from train import *
from utils import *
from FNN_model import *
from LSTM_model import *
from CNN_model import *
from MHAttention_model import *
from LSTMATTN_model import *

from torch.utils.data import TensorDataset, DataLoader

def hyperparameter_tunning(model_name,embeddings,embedding_type, params, num_param, current_combination, device, X, X_val, y_train, y_val, file_best_param,epochs):
  if num_param == len(params):
    
    #from string model_name to a class objet
    model = getattr(sys.modules[__name__], model_name)(embeddings, embedding_type,current_combination) 
    print(current_combination)
    train_loader = data_loaders(torch.Tensor(X), y_train, current_combination['batch_size'])
    val_loader = data_loaders(torch.Tensor(X_val), y_val, current_combination['batch_size'])
    optimizer = torch.optim.SGD(model.parameters(), lr = current_combination['learning_rate'])
    criterion = nn.BCELoss()
    model.to(device)
    losses = {'train': [],'validation' : []}
    accs = {'train' : [],'validation' : []}
    for e in range(epochs):
      if e % 1 == 0:
        print(e)
      val_loss, val_f, losses, accs = train(model, current_combination, train_loader, val_loader, device, criterion, optimizer, e, losses, accs)
      best_save_parameters = get_config(model_name, 'best', file_best_param)
      best_f1 = best_save_parameters["f_score"]
      if val_f > best_f1:

        print('current epoch', e ,'VAl f1 increased ({:.6f} --> {:.6f}).  Saving model ...'.format(best_f1,val_f))
        current_combination["best_epoch"] = e
        current_combination["f_score"] = val_f
        with open(file_best_param, 'w') as outfile:
            json.dump(current_combination,outfile )

    plot_training_performance(losses['train'],accs['train'],losses['validation'],accs['validation']) 
    del model
    return
  elif num_param < len(params):
    print("num_param", num_param)
    key = list(params.keys())[num_param]
    values = list(params.values())[num_param]
    # print(values)
    # print(key)
    for val in values:
      current_combination[key] = val
      hyperparameter_tunning(model_name,embeddings,embedding_type, params, num_param + 1, current_combination, device, X, X_val, y_train, y_val, file_best_param,epochs)