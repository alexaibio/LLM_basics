# Replicate the same transformer with PyTorch primitives

torch.nn has
```doctest
  ┌─────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────┐
  │         your class          │                                        built-in                                         │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ MultiHeadAttentionBlock     │ nn.MultiheadAttention, or F.scaled_dot_product_attention for just the attention formula │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ EncoderBlock / DecoderBlock │ nn.TransformerEncoderLayer / nn.TransformerDecoderLayer                                 │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ Encoder / Decoder           │ nn.TransformerEncoder / nn.TransformerDecoder                                           │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ the whole encoder–decoder   │ nn.Transformer                                                                          │
  ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ NormalizationLayer          │ nn.LayerNorm                                                                            │
  └─────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────┘
```

So no need to write it all manually except in learning purposes

  Things that differ from manual code:
  - Masks use the opposite meaning: in PyTorch, `True` means blocked, while in our code 1 means allowed. You'd pass src_key_padding_mask and
    tgt_mask (built with nn.Transformer.generate_square_subsequent_mask(S)).
  - Pass `batch_first=True`. Otherwise it expects (S, B, d).
  - Pass `norm_first=True` to match your pre-norm layout. The default is the paper's post-norm.

