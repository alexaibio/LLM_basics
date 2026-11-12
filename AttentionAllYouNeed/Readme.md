# 'Attention is all you need' replication
Encoder-decoded architecture for language translation.
Based on 2-language data train a foundational transformer on generative settings.

Шnspired by Umar Jamil 
https://www.youtube.com/watch?v=ISNdQcPhsts
https://github.com/hkproj/pytorch-transformer


# Abbreviation
- d_model - embeddings size
- h - number of head of attention


# Train pipeline
- get_ds(config) 
  - ds_raw = load_dataset (from haggingface.datasets) - ('opus_books', 'en-ru'), 17496x2
  - tokenizer_src = get_or_build_tokenizer, create tokenizer_en/ru.json
  - BilingualDataset
  - tokenizer_src.encode
  - return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
- dfdfd


# Main blocks
- model.py - low level PyTorch implementation
- model_short.py - high level PyTorch implementation (Encoder and decoder are built-in functions)

# TBD
- replicate with FastTransformers
- replicate with FlashAttention