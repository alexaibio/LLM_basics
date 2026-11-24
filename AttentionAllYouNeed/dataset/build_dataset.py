# Huggingface datasets and tokenizers
from datasets import load_dataset, load_dataset_builder
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader, random_split

from AttentionAllYouNeed.config import get_model_folder
from AttentionAllYouNeed.dataset.dataset import BilingualDataset


def _reshaped_dataset_gen(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def _get_or_build_tokenizer(config, ds, lang):
    """
    Create a WordLevel tokenizer (word=ID) which turns text into a list of integer IDs.
    Other HF tokenizers: BPE, WordPiece, Unigram
    """
    # saved next to the a_model_manual weights: .../AttentionAllYouNeed/model_saved/tokenizer_{lang}.json
    tokenizer_path = get_model_folder(config) / config['tokenizer_file'].format(lang)

    if not tokenizer_path.exists():
        # the simplest tokenizer = lookup table from word to ID
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))         # if tokenizer see a word not in vocabulary it substitutes it with UNK
        tokenizer.pre_tokenizer = Whitespace()                      # split sentence by white space

        # [SOS] - start of sentence / [EOS] - end of sentence
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)

        # The tokenizer splits every sentence, counts each word, drops
        #   the rare ones, and assigns IDs in order of frequency, most common first.
        tokenizer.train_from_iterator(_reshaped_dataset_gen(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    """
    Form BilingualDataset training and validation with corresponding tokenizers

    Output of ds[0]
      encoder_input  [ 2,  69, 1179,  3,  1,  1]   [SOS] Anna Karenina [EOS] [PAD] [PAD] - исходное предложение (анг)
      decoder_input  [ 2,  73,  628,  1,  1,  1]   [SOS] Анна Каренина [PAD] [PAD] [PAD] - только начало перевода (рус) - поэтому нет EOS
      label          [73, 628,    3,  1,  1,  1]   Анна Каренина [EOS] [PAD] [PAD] [PAD] - нет SOS - ответ, с которым сравнивается предсказание
      encoder_mask   (1, 1, 6)  [1, 1, 1, 1, 0, 0] - запрещает Encoder обращать внимание на PAD.
      decoder_mask   (1, 6, 6)  - запрещает Decoder смотреть на PAD и будущие токены.
      src_text, tgt_text        raw strings, used for printing and BLEU at validation

    """

    # Load huggingface Datasets: {'en': 'Anna Karenina', 'ru': 'Анна Каренина'}
    ds_raw = load_dataset(
        path=f"{config['datasource']}",                     # opus_books
        name=f"{config['lang_src']}-{config['lang_tgt']}",  # en-ru
        split='train'
    )

    # Build tokenizers
    tokenizer_src = _get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = _get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # out of HGF DataSets create a complete input/output tensors and a label tensor with masks
    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config['lang_src'], config['lang_tgt'], config['seq_len']
    )

    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config['lang_src'], config['lang_tgt'], config['seq_len']
    )

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


if __name__ == "__main__":
    from AttentionAllYouNeed.config import get_config
    conf = get_config()
    train_loader, val_loader, tok_src, tok_tar = get_ds(conf)