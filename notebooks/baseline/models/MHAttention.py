import torch.nn as nn

# from torch.nn import MultiheadAttention
class MHAttention(nn.Module):
    def __init__(self,embeddings,embedding_type, params):
        # (self, vocab_size, embed_dim, num_heads, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MHAttention, self).__init__()
        self.vocab_size = params["vocab_size"]
        self.n_layers = params["n_layers"]
        self.embedding_dim = params['embedding_dim']
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, sparse=True)
        if embedding_type != 'random':
            self.embedding.weight = nn.Parameter(embeddings)   
        self.embedding.weight = nn.Parameter(torch.tensor(embeddings, dtype=torch.float32))
        self.embedding.weight.requires_grad = params['trainable_embeddings']
        self.multihead_attn = MultiheadAttention(self.embed_dim, params['num_heads'])
        self.out = nn.Linear(self.embedding_dim, 1)

    def forward(self, x):
        x_prev = torch.zeros(x.shape)
        x = self.embedding(x.long())
        # attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        for i in range(self.n_layers):
            attn_output, attn_output_weights = self.multihead_attn(x, x, x)
            x = x_prev + attn_output
            x_prev = attn_output
        attn_output = torch.sum(attn_output, 1)
        # print(attn_output_weights.shape)
        output = self.out(attn_output)
        output = F.sigmoid(output)
        # print(attn_output.shape)
        return output