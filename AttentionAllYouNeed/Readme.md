# Three options to implement transformer
- a completely manual, `a_model_manual`
- with PyTorch primitives, `b_model_pytorch`
- with Hagging Face Library `c_model_hf`

Also, lets investigate how the model is loaded to HF transformer


# Option A: 'Attention is all you need' manual replication
Encoder-decoded architecture for language translation.
Based on 2-language data train a foundational transformer on generative settings.

References: 
- https://www.youtube.com/watch?v=ISNdQcPhsts
- https://github.com/hkproj/pytorch-transformer
- Vaswani etal, 2017. Attention is all you need. Advances in Neural Information Processing Systems, 30, 5998–6008.
- https://nlp.seas.harvard.edu/2018/04/03/attention.html


## Variables
- d_model - embeddings size
- h - number of head of attention


## Main blocks
- `data_process/` - build an appropriate dataset for training
- `model/model.py` - low level PyTorch Transformer implementation


## Dataset
- get_ds(config) 
  - ds_raw = load_dataset (from haggingface.datasets) - ('opus_books', 'en-ru'), 17496x2
  - tokenizer_src = get_or_build_tokenizer, create tokenizer_en/ru.json
  - BilingualDataset
  - tokenizer_src.encode
  - return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


## Simple Layers

### `InputEmbeddings`
- Learns embedding and places them into lookup table
- input x: indexes by which you can retrieve embedding `(batch, seq_len)`
- output of number of embedding with size d_model (`batch, seq_len, d_model)`


### `PositionalEncoding`
- we need to add to each word embedding information about its position in the sentence
- for that we build a vector of `d_model` size which carry positional information and add it to our embedding
- input x /output: `(batch, seq_len, d_model)`


### `MultiHeadAttentionBlock`
- input: q, k, v, mask
- output: embeddings for each wors (batch, seq_len, d_model)


### `LayerNormalization`

  In a Transformer, each block adds its output onto the residual stream (x + sublayer(...)). Stacked over many blocks, the values in each
  token's vector can drift: some grow large, some shrink, and their scale changes during training. That causes three problems:
  - Softmax saturates: large inputs to attention give almost one-hot weights with tiny gradients.
  - Training gets fragile: each layer's input keeps shifting as earlier layers change, so the learning rate has to be very small.
  - Deep stacks become unstable: values can explode or vanish.
 
Layer normalization resets every token's vector to a standard scale (mean 0, standard deviation 1) before each sublayer.
Each token is normalized separately, across its own 512 features.


### `ResidualConnection`

Inside each Attention block we use SkipConnections: we skip attention layer and add input to output of attention
it wraps one sublayer, either attention or the feed-forward block, and adds a skip path around it
- input: x, sublayer;   x: (B, seq_len, d_model)
- output: x;            x: (B, seq_len, d_model)


### `FeedForwardBlock`

position-wise feed-forward network from the paper (section 3.3).

One linear layer can't learn non-linear functions. A single Linear can only rotate, scale and shift the space. Putting ReLU between
  two of them makes the block a small neural network that can learn non-linear transformations.

    `FFN(x) = ReLU(x·W₁ + b₁)·W₂ + b₂`


### `ProjectionLayer`

It is the model's last step.
Project embedding back to the whole vocab size to assigns logits.
It's the largest single layer `512 × 15,178 + 15,178 ≈ 7.8M` parameters. That's about 28% of your whole 27.4M model, in one layer.

- `vocab_size` is the target vocabulary size (tokenizer_tgt.get_vocab_size()), which is 15,178 Russian tokens. 
- For each position, the output holds a logit score. 

logits are used everywhere:
  - Training: CrossEntropyLoss takes the logits of shape (B·S, vocab) and the label IDs of shape (B·S).
  - Inference: greedy_decode takes the logits at the last position, out[:, -1] with shape (1, vocab), and picks argmax.



## Model's Blocks

### `EncoderBlock` and `Encoder` 

- EncoderBlock: One block equals two sublayers (self-attention+Feed Forward), each wrapped in a ResidualConnection
- Encoder: Run N  EncoderBlocks with residial connections



### `DecoderBlock` and `Decoder`
TBD

### Finally: `Transformer` and `build_transformer`
 Transformer doesn't build anything itself. Its __init__ only stores parts that have already been built.
 The classes say what the model is, and the function says how to assemble one.

 Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

  Something has to create those seven parts, and that's build_transformer's job:
  1. Embeddings: two InputEmbeddings, for the source and target vocabularies.
  2. Positional encodings: two PositionalEncodings.
  3. Encoder: N encoder blocks, each with its own attention and feed-forward modules, in an nn.ModuleList.
  4. Decoder: N decoder blocks, each with self-attention, cross-attention and feed-forward modules.
  5. Projection: the ProjectionLayer.
  6. Initialization: Xavier initialization of all weight matrices (nn.init.xavier_uniform_). This step is easy to forget, and the model
     trains worse without it


## Training
`uv run python AttentionAllYouNeed/train_manual.py`

Metrics:
 - BLEU: it measures translation quality the way people judge it.
 - validation loos: prevent overfitting



## Inference
The general way to translate (decode) is as follows
 - Encoder - run once and build an internal representation of an input sentence
 - Decoder - run many times (always uses output from encoder), generate next token based on previous till the EOS.
   - Masked Self-Attention — смотрит на уже сгенерированные токены, не заглядывая в будущее.
   - Cross-Attention — обращается к выходу Encoder, чтобы учитывать исходное предложение.
   - NOTE: no KV cache here, на каждом шаге Decoder может заново обрабатывать всю предыдущую последовательность


Decoding can be done in several ways (see `decode.py`)
 - `greedy_decode()` - a simplest way when next token is just a token with highers probability
 - `beam_search_decode()` - a more complicated method
     

# Option B: with PyTorch primitives