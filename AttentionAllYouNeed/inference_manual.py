import sys

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from AttentionAllYouNeed.config import get_config, get_model_folder, latest_weights_file_path
from AttentionAllYouNeed.dataset.dataset import BilingualDataset
from AttentionAllYouNeed.a_model_manual.masks import padding_mask
from AttentionAllYouNeed.a_model_manual.model import build_transformer
from AttentionAllYouNeed.a_model_manual.decode import greedy_decode


def translate(sentence: str):
    """
    Decoding loop:
     encode once, then repeat: causal mask, then decode, then argmax, then append, until [EOS] or seq_len
    """
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(">>> Translate using device:", device)

    # tokenizers are saved next to the a_model_manual weights: .../AttentionAllYouNeed/model_saved/tokenizer_{lang}.json
    tokenizer_src = Tokenizer.from_file(str(get_model_folder(config) / config['tokenizer_file'].format(config['lang_src'])))
    tokenizer_tgt = Tokenizer.from_file(str(get_model_folder(config) / config['tokenizer_file'].format(config['lang_tgt'])))
    model = build_transformer(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Load the pretrained weights
    model_filename = latest_weights_file_path(config)
    if model_filename is None:
        raise FileNotFoundError(f"No checkpoint in {get_model_folder(config)}: train the model first (train.py)")
    state = torch.load(model_filename, map_location=device)   # map_location: a checkpoint saved on CUDA also loads on MPS/CPU
    model.load_state_dict(state['model_state_dict'])

    # instead writing the sentence you can use it by number from the dataset.
    label = ""
    if type(sentence) == int or sentence.isdigit():
        id = int(sentence)
        ds = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='all')
        ds = BilingualDataset(ds, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
        sentence = ds[id]['src_text']
        label = ds[id]["tgt_text"]
    seq_len = config['seq_len']

    # translate the sentence
    model.eval()
    with torch.no_grad():
        # Precompute the encoder output once and reuse it for every generation step
        source = tokenizer_src.encode(sentence)
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
        ], dim=0).to(device)
        source_mask = padding_mask(source, tokenizer_src.token_to_id('[PAD]'))   # (1, 1, seq_len)
        source = source.unsqueeze(0)        # add batch dimension: (seq_len) -> (1, seq_len)

        # Print the source sentence and target start prompt
        if label != "": print(f"{f'ID: ':>12}{id}")
        print(f"{f'SOURCE: ':>12}{sentence}")
        if label != "": print(f"{f'TARGET: ':>12}{label}")
        print(f"{f'PREDICTED: ':>12}", end='')

        # Generate the translation word by word (verbose=True prints each word as it is generated)
        model_out = greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, seq_len, device, verbose=True)

    # convert ids to tokens
    return tokenizer_tgt.decode(model_out.tolist())


if __name__ == "__main__":
    # read sentence from argument
    translate(sys.argv[1] if len(sys.argv) > 1 else "This is my first attempt to translate.")