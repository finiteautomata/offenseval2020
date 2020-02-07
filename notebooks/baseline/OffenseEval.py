import numpy as np
import argparse
import torch
np.random.seed(0)
import pandas as pd
from sklearn.model_selection import train_test_split
torch.random.manual_seed(0)
# from google.colab import drive
# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
is_cuda = torch.cuda.is_available()
# import fasttext

if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

import sys
sys.path.append('./src')
sys.path.append('./models')

from utils import *
from train import *

from hyperparameter_tunning import *


# from tunning import hyperparameter_tunning
# from tunning import hyperparameter_tunning
def Experimental_configuration(Text_train, y_train, Text_val, y_val, device, epochs):
  current_combination = {}
  #preparing data
  vocab = gen_vocab(Text_train)
  MAX_LEN =  max_length(Text_train)
  print(MAX_LEN)
  X = pad_sequences(gen_sequence(Text_train,vocab),  MAX_LEN)
  X_val = pad_sequences(gen_sequence(Text_val,vocab), MAX_LEN)
  embedding_type = 'glove'
  embeddings, embeddings_dim = load_embeddings(embedding_type, vocab)
  for model_name  in ['LSTM','LSTMATTN']: #'LSTMATTN' different models ['FNN', 'CNN', 'LSTM', 'LSTM_ATTN', 'MHAttention', 'LSTM_CNN']
      params = get_config(model_name,'config', None)
      for trainable_embeddings in [False]:

          current_combination["model_name"] = model_name
          # current_combination["embeddings"] = embeddings
          current_combination["max_len"] = 500
          # current_combination["device"] = device

          current_combination["vocab_size"] = len(vocab)
          current_combination["embedding_type"] = embedding_type
          current_combination["embedding_dim"] = embeddings_dim
          current_combination["trainable_embeddings"] = trainable_embeddings

          print(current_combination)
          file_best_param = "./logs/ENGLISH/best_param" + " " + current_combination["model_name"] + " " + current_combination["embedding_type"] + " " + "trainable " + str(current_combination["trainable_embeddings"]) + ".json" 
          with open(file_best_param, 'w') as outfile:
            json.dump({"f_score": 0}, outfile)
          hyperparameter_tunning(model_name, embeddings, embedding_type,params, 0, current_combination , device, X, X_val, y_train, y_val, file_best_param, epochs)

def average_to_label(y):
  y_temp = []
  for i in y:
    if i > 0.5:
      y_temp.append(1)
    else:
      y_temp.append(0)
  return y_temp

def main(epochs):

    data_train = pd.read_csv('../data_offenseEval/English/task_a_distant.train.tsv', sep='\t')


    data_val = pd.read_csv('../data_offenseEval/English/olid-training-v1.0.tsv', sep='\t')


    print(data_train.keys())
    print(len(data_train['text']))

    X_train = data_train['text']

    y_train = average_to_label(data_train['average'])


# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    vocab = gen_vocab(X_train)
    print(len(vocab))
    X_val = data_val['tweet']
    y_val = binarize(data_val['subtask_a'])
    print(Counter(y_val))
    print(Counter(y_train))
    Experimental_configuration(X_train, y_train, X_val, y_val, device, epochs)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter tunning')
    parser.add_argument('-e', '--epochs', required=True)
 
    args = parser.parse_args()
    epochs = int(args.epochs)
    main(epochs)
  
