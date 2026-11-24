# https://github.com/hkproj/pytorch-transformer/blob/main/model.py

import math
from collections.abc import Callable

import torch
import torch.nn as nn

################################################
#### Building blocks
################################################

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Learns embedding and places them into lookup table
        - d_model - length of embeddings vector
        - x: indexes by which you can retrieve embeding of size d_model
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # input x: (batch, seq_len) --> output (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings (according to the paper)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    add to each word embedding  a vector PE with information about word's position in the sentence
    (see exact formula in the article)
    Vector is different for even and odd word
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)          # placeholder for each word in input sequence

        # Create a vector of shape (seq_len) - делимое
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)

        # Create a vector of shape (d_model) - делитель для формулы
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))

        # Add a batch dimension to the positional encoding
        #  у нас будет много examples - и нам надо добавить dimension for that
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # requires_grad_(False) потому что мы не хотим учить позицию как параметр
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)      # (batch, seq_len, d_model)
        return self.dropout(x)


class NormalizationLayer(nn.Module):
    """
    Normalize the values in each feature across the sequence dimension.
    It's applied independently to each position in the sequence, which is crucial for the model's performance.
    """
    def __init__(self, features: int, eps: float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Each token is normalized separately, across its own 512 features:

        mean = x.mean(dim = -1, keepdim = True)     # (batch, seq_len, 1)

        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True)       # (batch, seq_len, 1)

        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """
    position-wise feed-forward network from the paper (section 3.3)
        FFN(x) = ReLU(x·W₁ + b₁)·W₂ + b₂
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)    # w1 and b1 (bias is True by default)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)    # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    """
    - h: number of heads
    """
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model      # Embedding vector size
        self.h = h                  # Number of heads
        assert d_model % h == 0, "d_model is not divisible by h"    # Make sure d_model is divisible by h

        self.d_k = d_model // h                             # Dimension of vector seen by each head

        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv

        self.w_o = nn.Linear(d_model, d_model, bias=False)     # Wo - weights for concatenation linear layer
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Attention for one HEAD: softmax[Q K^T / sqrt(d_k)]
        - d_k - it is not a dimension of embedding vector, it is size of one HEAD
        """
        d_k = query.shape[-1]       # size of one head (d_model / headы count)

        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # Write a very low value to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)  # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # return both adjusted vectors and attention scores (for vizualization)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # make q, k, v learnable by multiplying by weight matrix in Lenear layer
        query = self.w_q(q)     # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)       # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)     # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # split to multiple heads
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    """
    Skip connection - we skip attention layer and add input to output of attention
    it wraps one sublayer (attention or the feed-forward block) and adds a skip path around it
    - sublayer is a function
    """
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = NormalizationLayer(features)

    def forward(self, x, sublayer: Callable):
        """
        sublayer is a previous layer
        """
        return x + self.dropout(sublayer(self.norm(x)))


###############################################################
### Projection layer -- convert embeddings back to a vocabulary
###############################################################

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
