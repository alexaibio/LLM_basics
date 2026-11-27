# https://github.com/hkproj/pytorch-transformer/blob/main/model.py

import torch
import torch.nn as nn

from AttentionAllYouNeed.a_model_manual.layers import (
    FeedForwardBlock,
    InputEmbeddings,
    NormalizationLayer,
    MultiHeadAttentionBlock,
    PositionalEncoding,
    ProjectionLayer,
    ResidualConnection,
)

###########################################################################
### ENCODER / DECODER
###########################################################################

class EncoderBlock(nn.Module):
    """
    One block has two sub-layers, each wrapped in a ResidualConnection
    - d_model - dimensionality of embeddings
    x (B, S, 512)
   │
   ├─► Norm ─► Self-Attention(q=k=v=x, mask) ─► Dropout ─┐
   └──────────────────────────────────────────────── + ◄─┘   residual_connections[0]
   │
   ├─► Norm ─► FeedForward ─► Dropout ──────────────┐
   └──────────────────────────────────────────── + ◄┘       residual_connections[1]
   │
    x (B, S, 512)  ← same shape, enriched with context
    """
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block: MultiHeadAttentionBlock = self_attention_block
        self.feed_forward_block: FeedForwardBlock = feed_forward_block
        # An encoder block has two sub-layers, and each one gets its own residual connection:
        #   residual_connections[0]  →  wraps self-attention
        #   residual_connections[1]  →  wraps feed-forward
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # Form the whole encoder block:
        #   - attention_block wrapped in a residial conection
        #   - FF block, wrapped in residial
        #   - why lambda - ResidualConnection calls sublayer(self.norm(x)) with one argument, but attention needs four (q, k, v, mask)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(q=x, k=x, v=x, mask=src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x


class Encoder(nn.Module):
    """
    Run N EncoderBlocks
    Features = number of dimensions of embeddings
    """
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = NormalizationLayer(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)         # Note: Transformers use pre-norm plus one final norm.


class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 3 blocks: Masked MHA, MHA, FF
        # By using a lambda function, you can postpone the evaluation until the lambda function is called
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        '''
        features are size of embedding dimensions?
        '''
        super().__init__()
        self.layers = layers
        self.norm = NormalizationLayer(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)



###############################################
## Full TRANSFORMER
###############################################

class Transformer(nn.Module):
    """
    src_embedding - for input language
    tgt_embedding - for the target language

    We use separate encode and decode methods here (instead of forward) because we need Transformer in two modes
     - train (every time run both encoder and decoder)
     - inference (run encoder once, then only decoder)
    """
    def __init__(self,
                 encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer
                 ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # src - input text: batch, seq_len, d_model)
        # note: src_mask is not a causal mask, but padding mask, it comes from ds
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(x=src, mask=src_mask)   # forward() method of encoder() is called here

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.projection_layer(x)




##################################################################

def build_transformer_manual(config: dict, src_vocab_size: int, tgt_vocab_size: int) -> Transformer:

    # Hyperparameters come from config.py; vocab sizes come from the trained tokenizers
    src_seq_len = tgt_seq_len = config['seq_len']
    d_model = config['d_model']   # embedding size
    N = config['N']               # number of encoder blocks and of decoder blocks
    h = config['h']               # number of attention heads
    d_ff = config['d_ff']         # hidden size of the feed-forward block
    dropout = config['dropout']

    # 1 - Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # 2 - Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # 3 - Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # 4 - Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block,
                                     feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # 5 - Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))   # stores and registers encoder modules
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # 6 - Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    ############ FINAL: Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

