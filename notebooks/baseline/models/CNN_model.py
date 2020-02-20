import torch.nn as nn
import torch.nn.functional as F
import torch
class CNN(nn.Module):
    def __init__(self,embeddings,embedding_type, params):
        super(CNN, self).__init__()
        self.filter_size = params["filter_size"]
        self.num_filters = params["n_filters"]
        self.n_layers = params["n_layers"]
        self.embedding_dim = params['embedding_dim']

        self.name = 'CNN'

        self.embedding = nn.Embedding(params["vocab_size"] + 1, self.embedding_dim, sparse=True)
        if embedding_type != 'random':
            self.embedding.weight = nn.Parameter(embeddings)        
        self.embedding.weight.requires_grad = params['trainable_embeddings']

        # self.convs1 = nn.Conv2d(1, self.num_filters, (self.filter_size, embedding_dim))
        self.convs1 = nn.ModuleList([nn.Conv2d(1, self.num_filters, (self.filter_size, self.embedding_dim)) for i in range(self.n_layers)])

        self.fc1 = nn.Linear(self.num_filters * self.n_layers, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x.long())  
        x = x.unsqueeze(1)  
        # x = F.relu(self.convs1(x)).squeeze(3)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] 
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  

        # x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = torch.cat(x, 1)
        logit = self.fc1(x) 
        logit = self.sigmoid(logit)  
        return logit