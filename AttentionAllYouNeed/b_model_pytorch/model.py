# The same Transformer as a_model_manual, but built from PyTorch primitives (torch.nn).
# Same structure as a_model_manual/model.py:
#   build_transformer(config, src_vocab_size, tgt_vocab_size) -> model with .encode() / .decode() / .project()
# but it takes masks in the PyTorch convention (used by train_pytorch.py):
#   True = blocked, False = allowed        (a_model_manual: 1 = allowed, 0 = blocked)
#   - key padding masks (B, S): True where the token is [PAD]
#   - causal mask (L, L):       True above the diagonal (future positions)
#
#   a_model_manual                      b_model_pytorch
#   ------------------------------      ------------------------------------------
#   MultiHeadAttentionBlock          →  nn.MultiheadAttention (inside the layers below)
#   FeedForwardBlock                 →  part of nn.TransformerEncoderLayer / DecoderLayer
#   ResidualConnection               →  part of nn.TransformerEncoderLayer / DecoderLayer
#   NormalizationLayer               →  nn.LayerNorm
#   EncoderBlock / DecoderBlock      →  nn.TransformerEncoderLayer / nn.TransformerDecoderLayer
#   Encoder / Decoder (+ final norm) →  nn.TransformerEncoder / nn.TransformerDecoder (norm=nn.LayerNorm)
#   ProjectionLayer                  →  nn.Linear
#   InputEmbeddings, PositionalEncoding → no built-in, small versions below
#
# Small differences from a_model_manual (the model is equivalent, numbers are not identical):
#   - nn.MultiheadAttention uses bias in its Q/K/V/O projections (manual: bias=False)
#   - nn.LayerNorm uses the biased std and puts eps inside the sqrt (manual: unbiased std, eps added to std)
#   - attention uses PyTorch's fused kernels (scaled_dot_product_attention) when available

import math

import torch
import torch.nn as nn


################################################
#### Not provided by PyTorch
################################################

class InputEmbeddings(nn.Module):
    """Lookup table token id -> vector, scaled by sqrt(d_model) as in the paper."""
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len) --> (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding from the paper, added to the embeddings."""
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)                              # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))   # (d_model / 2)
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)   # even indices
        pe[:, 1::2] = torch.cos(position * div_term)   # odd indices
        self.register_buffer('pe', pe.unsqueeze(0))    # (1, seq_len, d_model), not trained, moves with .to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, d_model) --> same shape
        return self.dropout(x + self.pe[:, :x.shape[1], :])


###############################################
## Full TRANSFORMER
###############################################

class Transformer(nn.Module):
    def __init__(self, config: dict, src_vocab_size: int, tgt_vocab_size: int) -> None:
        super().__init__()
        d_model = config['d_model']
        assert d_model % config['h'] == 0, "d_model is not divisible by h"

        # 1 - Embeddings and 2 - positional encodings (same as manual)
        self.src_embed = InputEmbeddings(d_model, src_vocab_size)
        self.tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
        self.src_pos = PositionalEncoding(d_model, config['seq_len'], config['dropout'])
        self.tgt_pos = PositionalEncoding(d_model, config['seq_len'], config['dropout'])

        # Settings shared by encoder and decoder layers
        layer_args = dict(
            d_model=d_model,
            nhead=config['h'],
            dim_feedforward=config['d_ff'],
            dropout=config['dropout'],
            activation='relu',
            layer_norm_eps=1e-6,
            batch_first=True,   # tensors are (batch, seq_len, d_model), like in manual; default is (seq_len, batch, d_model)
            norm_first=True,    # pre-norm like manual ResidualConnection; default False = post-norm as in the paper
        )

        # 3 - Encoder: one EncoderBlock (self-attention + feed-forward, each with norm/dropout/residual) ...
        encoder_layer = nn.TransformerEncoderLayer(**layer_args)
        # ... copied N times + final norm (needed with pre-norm, same as manual Encoder)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config['N'], norm=nn.LayerNorm(d_model, eps=1e-6),
            enable_nested_tensor=False,   # the nested-tensor speedup is not available with norm_first=True
        )

        # 4 - Decoder: DecoderBlock (masked self-attention + cross-attention + feed-forward) x N + final norm
        decoder_layer = nn.TransformerDecoderLayer(**layer_args)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config['N'], norm=nn.LayerNorm(d_model, eps=1e-6))

        # 6 - Projection to vocabulary scores (logits)
        self.projection_layer = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        # src: token ids (B, S); src_key_padding_mask: (B, S), True = [PAD]
        x = self.src_pos(self.src_embed(src))                                      # (B, S, d_model)
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)          # (B, S, d_model)

    def decode(self, encoder_output: torch.Tensor, src_key_padding_mask: torch.Tensor,
               tgt: torch.Tensor, tgt_mask: torch.Tensor, tgt_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # tgt: token ids (B, L)
        # tgt_mask: causal mask (L, L), True = future; tgt_key_padding_mask: (B, L), True = [PAD] (None in greedy decoding)
        # PyTorch combines the two masks itself (manual version: one combined (B, 1, L, L) mask)
        x = self.tgt_pos(self.tgt_embed(tgt))                                      # (B, L, d_model)
        return self.decoder(
            x,
            memory=encoder_output,                              # keys/values for cross-attention
            tgt_mask=tgt_mask,                                  # decoder self-attention: no looking at the future
            tgt_key_padding_mask=tgt_key_padding_mask,          # decoder self-attention: ignore Russian [PAD]
            memory_key_padding_mask=src_key_padding_mask,       # cross-attention: ignore English [PAD]
        )                                                       # (B, L, d_model)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, d_model) --> (B, L, vocab_size)
        return self.projection_layer(x)


##################################################################

def build_transformer_pytorch(config: dict, src_vocab_size: int, tgt_vocab_size: int) -> Transformer:
    transformer = Transformer(config, src_vocab_size, tgt_vocab_size)

    # Initialize the parameters (same as manual).
    # Also needed because nn.TransformerEncoder/Decoder deep-copy one layer N times,
    # so all N layers would otherwise start with identical weights.
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
