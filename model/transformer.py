import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import os


class Norm(nn.Module):
    """Normalisation layer"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
    """Attention function"""
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    scores_ = scores
    if dropout is not None:
        scores_ = dropout(scores_)
    output = torch.matmul(scores_, v)
    return output, scores


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer"""
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        # transpose to get dimensions bs * N_heads * seq_len * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention
        output, scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        att = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(att)

        return output, [att, scores]


class FeedForward(nn.Module):
    """Feed-Forward layer"""
    def __init__(self, d_model, hidden_size=2048, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_size, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class EncoderLayer(nn.Module):
    """Encoder layer"""
    def __init__(self, d_model, heads, dropout=0.1, hidden_size=2048):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, hidden_size, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm_1(x)
        output, scores = self.attn(x2, x2, x2, mask)
        x = x + self.dropout_1(output)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x, scores


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    """Transformer encoder"""
    def __init__(self, d_model, N, heads, dropout=0.1, hidden_size=2048):
        super().__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout, hidden_size), N)
        self.norm = Norm(d_model)

    def forward(self, x, mask=None):
        scores = [None] * self.N
        for i in range(self.N):
            x, scores[i] = self.layers[i](x, mask)
        return self.norm(x), scores


class Transformer(nn.Module):
    """Transformer"""
    def __init__(self, n_items, d_model, N, heads, dropout=0.1, hidden_size=2048):
        super().__init__()
        self.encoder = Encoder(d_model, N, heads, dropout, hidden_size)
        self.out = nn.Linear(d_model, n_items)

    def forward(self, x, mask=None, get_embedding=False, get_scores=False):
        assert get_embedding + get_scores < 2
        x_embedding, scores = self.encoder(x, mask)
        x = torch.mean(x_embedding, dim=-2)
        output = self.out(x)
        if get_embedding:
            return output, x_embedding
        elif get_scores:
            return output, scores
        else:
            return output


def get_model(n_items, d_model, heads=5, dropout=0.5, n_layers=6, hidden_size=2048, weights_path=None, device="cpu"):
    assert d_model % heads == 0
    assert dropout < 1

    model = Transformer(n_items, d_model, n_layers, heads, dropout, hidden_size)

    if weights_path is not None:
        if not weights_path.endswith('.pth'):
            weights_path = os.path.join(weights_path, "weights.pth")
        print("loading pretrained", weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    else:  # init weights using xavier
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    model = model.to(device)

    return model
