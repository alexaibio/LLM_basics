"""
Inference for both models with shared helpers.
    translate_manual()  - a_model_manual  (greedy or beam search)
    translate_pytorch() - b_model_pytorch (greedy)

Run from the repo root:
    uv run python -m AttentionAllYouNeed.inference
"""
import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from AttentionAllYouNeed.config import get_config, get_model_folder, latest_weights_file_path


################################################
#### Shared helpers
################################################

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def load_tokenizers(config) -> tuple[Tokenizer, Tokenizer]:
    # tokenizers are saved next to the model weights: .../model_saved/{manual|pytorch}/tokenizer_{lang}.json
    folder = get_model_folder(config)
    if not folder.exists():
        raise FileNotFoundError(f"No folder {folder}: train the model first")
    tokenizer_src = Tokenizer.from_file(str(folder / config['tokenizer_file'].format(config['lang_src'])))
    tokenizer_tgt = Tokenizer.from_file(str(folder / config['tokenizer_file'].format(config['lang_tgt'])))
    return tokenizer_src, tokenizer_tgt


def load_checkpoint(model: torch.nn.Module, config, device: torch.device) -> None:
    # Load the latest trained weights into the model
    model_filename = latest_weights_file_path(config)
    if model_filename is None:
        raise FileNotFoundError(f"No checkpoint in {get_model_folder(config)}: train the model first")
    state = torch.load(model_filename, map_location=device)   # map_location: a checkpoint saved on CUDA also loads on MPS/CPU
    model.load_state_dict(state['model_state_dict'])
    model.eval()


def get_sentence(sentence: str | int, config) -> tuple[str, str]:
    # Instead of typing a sentence you can pass a number: index of a sentence in the dataset.
    # Returns (source sentence, reference translation or "" if none)
    if type(sentence) == int or sentence.isdigit():
        ds = load_dataset(config['datasource'], f"{config['lang_src']}-{config['lang_tgt']}", split='train')
        pair = ds[int(sentence)]['translation']
        return pair[config['lang_src']], pair[config['lang_tgt']]
    return sentence, ""


def prepare_source(sentence: str, tokenizer_src: Tokenizer, seq_len: int, device: torch.device) -> torch.Tensor:
    # Text -> fixed-size tensor of token ids, same format as encoder_input in BilingualDataset:
    #   [SOS] w1 w2 ... wn [EOS] [PAD] ... [PAD]   -> (1, seq_len)
    enc = tokenizer_src.encode(sentence)
    assert len(enc.ids) <= seq_len - 2, f"Sentence is too long: {len(enc.ids)} tokens, max is {seq_len - 2}"
    source = torch.cat([
        torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),
        torch.tensor(enc.ids, dtype=torch.int64),
        torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
        torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(enc.ids) - 2), dtype=torch.int64),
    ], dim=0).to(device)
    return source.unsqueeze(0)   # add batch dimension: (seq_len) -> (1, seq_len)


def print_header(title: str, sentence: str, label: str) -> None:
    print(f"\n>>> {title}")
    print(f"{'SOURCE: ':>12}{sentence}")
    if label != "": print(f"{'TARGET: ':>12}{label}")
    print(f"{'PREDICTED: ':>12}", end='')


################################################
#### Model-specific translate functions
################################################

def translate_manual(sentence: str | int, method: str = "greedy") -> str:
    from AttentionAllYouNeed.a_model_manual.model import build_transformer
    from AttentionAllYouNeed.a_model_manual.masks import padding_mask
    from AttentionAllYouNeed.a_model_manual.decode import greedy_decode, beam_search_decode

    config = get_config()
    config['model_folder'] = f"{config['model_folder']}/manual"     # same as train_manual.py
    device = get_device()

    tokenizer_src, tokenizer_tgt = load_tokenizers(config)
    model = build_transformer(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    load_checkpoint(model, config, device)

    sentence, label = get_sentence(sentence, config)
    with torch.no_grad():
        source = prepare_source(sentence, tokenizer_src, config['seq_len'], device)                 # (1, seq_len)
        source_mask = padding_mask(source[0], tokenizer_src.token_to_id('[PAD]'))                     # (1, 1, seq_len), 1 = allowed

        print_header(f"manual, {method}", sentence, label)
        if method == "greedy":
            model_out = greedy_decode(model, source, source_mask, tokenizer_tgt, config['seq_len'], device, verbose=True)
            print()
        elif method == "beam":
            model_out = beam_search_decode(model, source, source_mask, tokenizer_tgt, config['seq_len'], device)
            print(tokenizer_tgt.decode(model_out.tolist()))
        else:
            raise ValueError(f"Unknown method: {method!r}, use 'greedy' or 'beam'")

    return tokenizer_tgt.decode(model_out.tolist())


def translate_pytorch(sentence: str | int) -> str:                  # different from manual: no method, greedy only
    from AttentionAllYouNeed.b_model_pytorch.model import build_transformer     # different from manual: b_model_pytorch
    from AttentionAllYouNeed.b_model_pytorch.decode import greedy_decode        # different from manual: b_model_pytorch, no beam search
    # different from manual: no masks import, the padding mask is built inline below

    config = get_config()
    config['model_folder'] = f"{config['model_folder']}/pytorch"    # different from manual: same as train_pytorch.py
    config['model_basename'] = 'pytorch_transl_model_'              # different from manual: same as train_pytorch.py
    device = get_device()

    tokenizer_src, tokenizer_tgt = load_tokenizers(config)
    model = build_transformer(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    load_checkpoint(model, config, device)

    sentence, label = get_sentence(sentence, config)
    with torch.no_grad():
        source = prepare_source(sentence, tokenizer_src, config['seq_len'], device)                 # (1, seq_len)
        src_pad_mask = source == tokenizer_src.token_to_id('[PAD]')                                   # different from manual: (1, seq_len), True = [PAD]

        print_header("pytorch, greedy", sentence, label)                                              # different from manual: title
        model_out = greedy_decode(model, source, src_pad_mask, tokenizer_tgt, config['seq_len'], device)   # different from manual: PyTorch mask, no verbose
        print(tokenizer_tgt.decode(model_out.tolist()))                                               # different from manual: prints at the end, not word by word

    return tokenizer_tgt.decode(model_out.tolist())


if __name__ == "__main__":
    test_sentence = "Who is Levin?"

    translate_manual(test_sentence, method="greedy")
    translate_manual(test_sentence, method="beam")

    translate_pytorch(test_sentence)
