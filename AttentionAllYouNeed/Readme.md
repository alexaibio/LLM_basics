# 'Attention is all you need' replication
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


## Layers and building blocks
`InputEmbeddings`
- 


# TBD
- replicate with FastTransformers
- replicate with FlashAttention