import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class BilingualDataset(Dataset):
    """
    Turns one English–Russian text pair into the fixed-size tensors a Transformer trains on.
    It does:
        - create a fixed size training vectors with "seq_len": 350
        - takes a pair of parallel sentences (e.g., English → Russian), tokenizes them, adds special tokens ([SOS], [EOS], [PAD])
        - Labels are not given to the model directly as inputs. Instead, they are used in the loss function.
        - Teacher Forcing: почему Label сдвинут вперед по отношению к Decoder? - чтобы предсказывать след слово

    Input
      {'translation': {'en': 'Anna Karenina', 'ru': 'Анна Каренина'}}
      Token IDs: Anna=69, Karenina=1179, Анна=73, Каренина=628. Specials: [PAD]=1, [SOS]=2, [EOS]=3.

    Output of ds[0]
      encoder_input  [ 2,  69, 1179,  3,  1,  1]   [SOS] Anna Karenina [EOS] [PAD] [PAD] - исходное предложение (анг)
      decoder_input  [ 2,  73,  628,  1,  1,  1]   [SOS] Анна Каренина [PAD] [PAD] [PAD] - только начало перевода (рус) - поэтому нет EOS
      label          [73, 628,    3,  1,  1,  1]   Анна Каренина [EOS] [PAD] [PAD] [PAD] - нет SOS - ответ, с которым сравнивается предсказание
      encoder_mask   (1, 1, 6)  [1, 1, 1, 1, 0, 0] - запрещает Encoder обращать внимание на PAD.
      decoder_mask   (1, 6, 6)  - запрещает Decoder смотреть на PAD и будущие токены.
      src_text, tgt_text        raw strings, used for printing and BLEU at validation

    Special tokens
     - SOS = 2, 	Начало предложения
     - EOS = 3, 	Конец предложения
     - PAD = 1, 	Заполнение пустых позиций
    """

    def __init__(self, ds, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)  # 2
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)  # 3
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)  # 1

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2   # encoder adds [SOS] and [EOS] → 2 extra
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1   # decoder adds only [SOS]  → 1 extra

        # Make sure the number of padding tokens is not negative.
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # assemble input vector: Add SOS and EOS token, padding: [SOS] w1 w2 ... wn [EOS] [PAD] [PAD] ...]
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token: [SOS] w1 w2 ... wn [PAD] [PAD] ...]
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token: [w1 w2 ... wn [EOS] [PAD] [PAD] ...]
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)

            # encoder_mask: padding only: (1, 1, seq_len),  [[[1, 1, 1, 1, 0, 0]]]
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),

            # decoder_mask: padding AND causal: (1, seq_len) & (1, seq_len, seq_len),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,                 # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    '''
    This is crucial in autoregressive models where the prediction of each position should only depend on previous positions.
    - we dont need PAD tokens to be seen by self-attention mechanism
    - we dont need every word to see words coming after it in a sentence
    '''
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

