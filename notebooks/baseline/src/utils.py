
# coding: utf-8
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import os
import gensim
import pickle
from gensim.models.wrappers import FastText
from collections import Counter
import sys
sys.path.append('/gdrive/My Drive/offenseval2020-master/code/models')
import FNN_model
from FNN_model import FNN
# from Models import HateBiLSTM,HateCNN,HateLSTM,HateFNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn
torch.manual_seed(1)
np.random.seed(1)
import json
import gensim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from nltk import tokenize as tokenize_nltk
from string import punctuation
import gensim
from gensim.utils import tokenize
from gensim.parsing.preprocessing import STOPWORDS
from keras.preprocessing.sequence import pad_sequences
# !pip install fasttext
# import fasttext

TOKENIZER = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize
# from gensim.models import FastText as ft
def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id

def load_embeddings(vec, vocab):
  print("Loading embeddings ...")
  word2idx = {word: ii for ii, word in enumerate(vocab, 1)}
  if vec == 'glove':
    path = '../../../Vectores/glove.txt'
    # Workspace\Code\offenseval2020-master\code\src
    embedding_dim = 200
    with open(path, 'r', encoding = 'utf-8') as f:
        embeddings = np.zeros((len(word2idx) + 1, embedding_dim))
        c = 0 
        if c< 40000:
          for line in f.readlines():
              values = line.split()
              word = values[0]
              index = word2idx.get(word)
              if index:
                  try:
                    vector = np.array(values[1:], dtype='float32')
                    embeddings[index] = vector
                  except:
                    pass
              c += 1
    return torch.from_numpy(embeddings).float(), embedding_dim

  elif vec == 'w2v':
        path = '../../../Vectores//word2vec_twitter_tokens.bin'
        model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True, encoding='utf-8', unicode_errors='ignore')
        embedding_dim = 400

  elif vec == 'ft':
        path = '../../../Vectores/ft_300.bin'
        embedding_dim = 100
        # Load Skipgram or CBOW model
        model = fasttext.load_model(path)
        # model=ft.load_fasttext_format(path)

  else:
        flag = 'Muse'
        path  = '../../../Vectores/wiki.multi.' + vec +'.vec.txt'
        model, id2word, word2id = load_vec(path, 50000)
        embedding_dim = 300
        # Load Skipgram or CBOW model
        # model=ft.load_fasttext_format(path)

  embeddings = np.zeros((len(word2idx) + 1, embedding_dim))
  c = 0 
  for k, v in vocab.items():
      # print(k,v)
      # print(model[k])
      try:
        if flag == 'Muse':
          embeddings[v] = model[word2id[k]]
 
        else:
          embeddings[v] = model[k]
        # print('siiiii')
      except:
        pass
  del model
  return torch.from_numpy(embeddings).float(), embedding_dim

def mult(lista, entero):
    for i in range(len(lista)):
        lista[i] *= entero
    return lista

def gen_vocab(tweets):
    vocab, reverse_vocab = {}, {}
    vocab_index = 1
    for tweet in tweets:
        text = tokenize(tweet.lower())
        text = ' '.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]
        for word in words:
            if word not in vocab:
                vocab[word] = vocab_index
                reverse_vocab[vocab_index] = word       # generate reverse vocab as well
                vocab_index += 1
    vocab['UNK'] = len(vocab) + 1
    reverse_vocab[len(vocab)] = 'UNK'
    return vocab


def gen_sequence(tweets,vocab):
    X = []
    for tweet in tweets:
        text = tokenize(strip_handles=True, reduce_len=True).tokenize(tweet.lower())
        # text = ' '.join([c for c in text if c not in punctuation])
        # words = text.split()
        # words = [word for word in words if word not in STOPWORDS]
        seq, _emb = [], []
        for word in text:
            seq.append(vocab.get(word, vocab['UNK']))
        X.append(seq)
    return X  


def plot_training_performance(loss, acc, val_loss, val_acc):
    plt.figure(figsize=(5,5))
    plt.title("loss")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['train', 'validation'], loc='upper left')

    plt.figure(figsize=(5,5))
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def get_config(model_name, var, json_file):
    #read parameters form config file
    configurations = 1
    if var == 'config':
        json_file = open('../config/' + model_name + '.json', 'r')
        params = json.load(json_file)
        for value in params.values():
            configurations *= len(value)
        print("The posible configurations are: ", configurations)
        return(params)
    if var == 'best':
        best = open(json_file, 'r')
        params = json.load(best)
        return(params)

def data_loaders(X, y_train, batch_size):
 
  y_train = np.array(binarize(y_train))  
  train_data = TensorDataset(X, torch.tensor(y_train))
  train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
  return train_loader

from sklearn.model_selection import train_test_split

def max_length(X):
  post_length = max(np.array([len(x.split(" ")) for x in X]))
  return post_length 

def save_object(obj, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)

def load_object(filename):
    with open(filename, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def load_data(dataset):
    data = load_object(dataset)
    x_text = []
    labels = []
    for i in range(len(data)):
        x_text.append(data[i]['text'])
        labels.append(data[i]['label'])
    return x_text, labels

def binarize(y_):
    y_map = {
            'none': 0,
            'normal': 0,
            'neither':0,
             'both':1,
            'racism': 1,
            'sexism': 1,
            'hate':1,
            'hateful':1,
            'abusive': 1,
            'ofenssive':1,
            1: 1,
            0:0
    }
    y = []
    for i in y_:
        y.append(y_map[i])
    return y  
  
def save_object(obj, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)

def load_object(filename):
    with open(filename, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def load_data(dataset):
    data = load_object(dataset)
    x_text = []
    labels = []
    for i in range(len(data)):
        x_text.append(data[i]['text'])
        labels.append(data[i]['label'])

    print(Counter(x_text))
    print(Counter(labels))
    return x_text, labels

#MAX LENGTH FUNCTION
#FUNCTION TO CALCULATE THE MAX LENGTH OF THE DATA TO BE USE FOR PADDING LATER
def max_length(X):
  post_length = max(np.array([len(x.split(" ")) for x in X]))
  return post_length 


#FUNCTION TO MAP INTO BINARY CLASSES
#THE ORGINAL CLASSES ARE TANSFORMED INTO BINARY 
def binarize(y_):
    y_map = {
            'none': 0,
            'normal': 0,
            'neither':0,
            'NOT':0,
            'both':1,
            'OFF':1,
            'racism': 1,
            'sexism': 1,
            'hate':1,
            'hateful':1,
            'abusive': 1,
            'ofenssive':1,
            1: 1,
            0:0
    }
    y = []
    for i in y_:
        y.append(y_map[i])
    return y  

#EVALUATION METRICS FUNCTIONS
import re
from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def evaluate(predictions, test_labels):
    precision, recall, fscore, support = score(test_labels, predictions)
    accuracy = accuracy_score(test_labels, predictions)
    print('accuracy: {}'.format(accuracy))
    print('precision: {}'.format(precision))
    print(sum(precision)/2)
    print('recall: {}'.format(recall))
    print(sum(recall)/2)
    print('fscore: {}'.format(fscore))
    print(sum(fscore)/2)
    print('support: {}'.format(support))

def plot_training_performance(loss, acc, val_loss, val_acc):
    plt.figure(figsize=(5,5))
    plt.title("loss")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['train', 'validation'], loc='upper left')

    plt.figure(figsize=(5,5))
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def get_embedding_weights(vectors_file):
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(vectors_file)
    embedding = torch.zeros((len(vocab) + 1, EMBEDDING_DIM))
    n = 0
    for k, v in vocab.iteritems():
        try:
            embedding[v] = word2vec_model[k]
        except:
            n += 1
            pass
    
    return embedding


def finalizar_trainning(losses,accs):
    plot_training_performance(losses,accs,'losses')
    plot_training_performance(losses,accs,'accs')

def gen_sequence(tweets,vocab):
    X = []
    for tweet in tweets:
        text = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(tweet.lower())
        text = ' '.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]
        seq, _emb = [], []
        for word in text:
            seq.append(vocab.get(word, vocab['UNK']))
        X.append(seq)
    return X  
