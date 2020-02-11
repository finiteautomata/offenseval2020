import torch.nn as nn

class BERTGRUSequenceClassifier(nn.Module):

    """
    BERT + GRU model

    Inspired on Ben Trevett's implementation:
    https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/6%20-%20Transformers%20for%20Sentiment%20Analysis.ipynb
    """
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers=1,
                 bidirectional=False,
                 finetune_bert=False,
                 dropout=0.2):

        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']

        self.finetune_bert = finetune_bert

        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)

        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        #text = [batch size, sent len]

        if not self.finetune_bert:
            with torch.no_grad():
                embedded = self.bert(text)[0]
        else:
            embedded = self.bert(text)[0]
        #embedded = [batch size, sent len, emb dim]
        _, hidden = self.rnn(embedded)

        #hidden = [n layers * n directions, batch size, emb dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])

        #hidden = [batch size, hid dim]

        output = self.out(hidden)

        #output = [batch size, out dim]

        return output
