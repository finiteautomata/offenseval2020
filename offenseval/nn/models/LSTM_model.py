import torch.nn as nn
import torch
class LSTM(nn.Module):
    def __init__(self,embeddings,embedding_type, params): 
        super(LSTM, self).__init__()
        self.name = 'LSTM'
        self.n_layers = params['n_layers']
        self.mode = params['modes']
        self.hidden_dim = params['hidden_dim']
        self.bidirectional = bool(params['bidirectional'])
        self.embedding_dim = params['embedding_dim']
        self.directions = 1
        if self.bidirectional:
            self.directions *= 2

        self.embedding = nn.Embedding(params['vocab_size'] + 1, self.embedding_dim, sparse=True)
        if embedding_type != 'random':
            self.embedding.weight = nn.Parameter(embeddings)
        
        self.embedding.weight.requires_grad = params['trainable_embeddings']

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.n_layers ,bidirectional = self.bidirectional, batch_first=True)

        self.fc = nn.Linear(self.hidden_dim * self.directions, 1)
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.long()
        lstm_out, hidden = self.lstm(self.embedding(x))
        
        if self.mode == 'max':
          lstm_out = torch.max(lstm_out, 1)[0]
        elif self.mode == 'sum':
          lstm_out = torch.sum(lstm_out, 1)
        elif self.mode == 'avg':
          lstm_out = torch.mean(lstm_out.contiguous().view(-1, self.hidden_dim * self.directions), 1)    
        elif self.mode == 'last':   
          lstm_out = hidden
        else:
          lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim* self.directions)

        out = torch.sigmoid(self.fc(lstm_out))
        out = out.view(batch_size, -1)
        out = out[:,-1]
        return out
 