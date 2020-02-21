import torch.nn as nn
import torch
class Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dec_units, enc_units, batch_sz, heads, bi, params):
        super(Attention, self).__init__()

        self.batch_sz = params["batch_size"]
        self.heads = params['num_heads']
        self.dec_units = params["hidden_dim"]
        self.enc_units = params["hidden_dim"]
        self.vocab_size = params["vocab_size"]
        self.embedding_dim = params['embedding_dim']
        self.bidirectional = bool(params['bidirectional'])
        self.directions = 1
        if self.bidirectional:
            self.directions *= 2
        self.fc = nn.Linear(self.dec_units, 1)
        # used for attention
        self.W1 = nn.Linear(self.directions * self.enc_units, self.dec_units)
        self.W2 = nn.Linear(self.directions * self.dec_units, self.dec_units)
        self.V = nn.Linear( self.enc_units, self.dec_units)
    
    def forward(self, hidden, enc_output):
 
        hidden_with_time_axis = hidden.permute(1, 0, 2) 
        
        # print('enc_output.shape')
        # print(hidden.shape)   
        # print(enc_output.shape)   


        hidden = hidden.view(self.batch_sz, -1 ,self.directions * self.dec_units)

        # print('self.W1(enc_output).shape')
        # print(self.W1(enc_output).shape)  
        # print(self.W2(hidden).shape)

        q = self.W1(enc_output)#.view(self.batch_sz, enc_output.shape[1], self.heads, self.enc_units)

        # hidden = hidden.permute(1, 0, 2) 

        
        k = self.W2(hidden)#.view(self.batch_sz, -1, self.heads, self.enc_units)

        # print('q , k') 
        # print(q.shape) 
        # print(k.shape)
        
        # q = q.permute(2, 0, 1, 3).contiguous()#.view(-1, enc_output.shape[1], self.enc_units) # (n*b) x lq x dk

        # k = k.permute(2, 0, 1, 3).contiguous()#.view(-1, hidden.shape[1], self.enc_units) # (n*b) x lk x dk


        # print('q , k') 
        # print(q.shape) 
        # print(k.shape) 

        # if self.directions > 1:
        #   q = q.view(1, q.shape[0], enc_output.shape[1], self.enc_units * self.heads) 
        #   k = k.view(2, k.shape[0], 1, self.enc_units * self.heads)
        # # print('q , k') 
        # print(q.shape) 
        # print(k.shape)   
        K = torch.tanh(q + k)
        # except:
        #     print('q , k') 
        #     print(q.shape) 
        #     print(k.shape) 

        K = K#.view(self.batch_sz * self.heads, enc_output.shape[1], self.directions * self.enc_units)
 
        try:
            score = self.V(K)
        except:
            print('k.shape')
            print(K.shape)        # print('score.shape')
        # print(score.shape)

        attention_weights = torch.softmax(score, dim=1) # alpha


        # print('attention_weights') 
        # print(attention_weights.view(self.batch_sz,-1,42).shape) 
        # print('enc_output.shape') 
        # print(enc_output.shape) 
        # print(attention_weights.shape) 


        # print(enc_output.view(-1, enc_output.shape[0], enc_output.shape[1], self.enc_units).shape) 

        # torch.einsum('ijk,abk->abc', (attention_weights, enc_output))

        # context_vector = enc_output * attention_weights
        
        attention_weights = attention_weights.permute(0, 2, 1)

        context_vector = torch.bmm(attention_weights, enc_output)
        context_vector = context_vector.view(self.batch_sz,-1,self.enc_units)
        
        # context_vector = attention_weights.bmm(enc_output)

        # print('context_vector.shape')
        # print(context_vector.shape)

        context_vector = torch.sum(context_vector, dim=1) # a

        # print('context_vector.shape')
        # print(context_vector.shape)

        
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        # # takes case of the right portion of the model above (illustrated in red)
        # x = self.embedding(x)
    
        # x = torch.cat((context_vector.unsqueeze(1), x), -1)
    
        x = self.fc(context_vector)
        
        return torch.sigmoid(x), attention_weights
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        hidden = (weight.new(1, batch_size, self.hidden_dim).zero_().to(device),
          weight.new(1, batch_size, self.hidden_dim).zero_().to(device))
        return hidden



class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embeddings, bi, hidden_dim, batch_sz,params):
            
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.bidirectional = bool(bi)
        self.n_layers = 1
        self.directions = 1
        if self.bidirectional:
            self.directions *= 2
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(params['vocab_size'] + 1, self.embedding_dim, sparse=True)
        if params['embedding_type'] != 'random':
            self.embedding.weight = nn.Parameter(embeddings)
        
        self.embedding.weight.requires_grad = params['trainable_embeddings']

        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, bidirectional = self.bidirectional, batch_first=True)

    def forward(self, x):
        x = self.embedding(x.long()) 
        # self.hidden = self.init_hidden(self.batch_sz)
        output, self.hidden = self.lstm(x)         
        return output, self.hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers * self.directions, batch_size, self.hidden_dim).zero_().to(device),
          weight.new(self.n_layers * self.directions, batch_size, self.hidden_dim).zero_().to(device))

        return hidden

class LSTMATTN(nn.Module):
    def __init__(self,embeddings,embedding_type, params):
        super(LSTMATTN, self).__init__()
        self.name = 'LSTMATTN'
        self.max_len = params["max_len"]
        # self.device = params["device"]
        self.encoder = Encoder(params["vocab_size"], params["embedding_dim"], embeddings,params['bidirectional'], params["hidden_dim"], params["batch_size"],params)
        self.decoder = Attention(params["vocab_size"], params["embedding_dim"], params["hidden_dim"], params["hidden_dim"], params["batch_size"], params["num_heads"], params['bidirectional'],params)
    def forward(self, inputs):
          rnn_output, enc_hidden = self.encoder(inputs)
        #   enc_hidden = torch.mean(enc_hidden[0], 0 )

          predictions, _ = self.decoder(enc_hidden[0], rnn_output)
          return predictions