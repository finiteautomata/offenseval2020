import torch.nn as nn
import torch
class FNN(nn.Module):
    def __init__(self,embeddings,embedding_type, params):
        super(FNN, self).__init__()
        self.params = params
        self.name = 'FNN'
        self.n_layers = params['n_layers']
        self.hidden_dim = params['hidden_dim']
        self.embedding_dim = params['embedding_dim']
        self.cmode = params['modes']

        self.embedding = nn.Embedding(params['vocab_size'] + 1, self.embedding_dim)
        if embedding_type != 'random':
          self.embedding.weight = nn.Parameter(embeddings)
        
        self.embedding.weight.requires_grad = params['trainable_embeddings']
        self.fc1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 1)
        self.fc = nn.Linear(self.embedding_dim, 1)
        # self.dropout = nn.Dropout(drop_prob)
                
        self.sig = nn.Sigmoid()

    def avg_embeddings(self, text):
        avg_text = torch.Tensor(text.shape[0],self.embedding_dim)
        for t in range(len(text)):
          tt = torch.sum(text[t])/text.shape[1]
          avg_text[t] = tt
        return avg_text.to(device)
        
    def forward(self, x):
        embedded = self.embedding(x.long())
        if self.cmode == 'max':
          embedded = torch.max(embedded, 1)[0]
        elif self.cmode == 'sum':
          embedded = torch.sum(embedded, 1)
        else:
          embedded = torch.mean(embedded, 1)

        if self.n_layers == 1:
          linear = self.fc(embedded) 
        elif self.n_layers == 2:
          linear = self.fc1(embedded) 
          linear = self.fc2(linear) 
        sig_out = self.sig(linear)
        return sig_out
