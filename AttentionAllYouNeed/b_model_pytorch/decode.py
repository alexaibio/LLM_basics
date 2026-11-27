import torch
from tokenizers import Tokenizer

from AttentionAllYouNeed.b_model_pytorch.masks import causal_mask
from AttentionAllYouNeed.b_model_pytorch.model import Transformer


def greedy_decode(
        model: Transformer,
        source: torch.Tensor,           # (1, seq_len)
        src_pad_mask: torch.Tensor,     # (1, seq_len), True = [PAD], INVERTED
        tokenizer_tgt: Tokenizer,
        max_len: int,
        device: torch.device,
) -> torch.Tensor:                      # (generated_len,)
    """
    for inference (generation) we have to run encoder only once!
    - compute encoder once, then generate first token by decoder, then add it to decoded input etc
    - generation stops when we reach EOS or max_length
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, src_pad_mask)

    # Initialize the decoder input with the sos token (1, 1) = batch x first token
    decoder_input = torch.tensor([[sos_idx]], dtype=source.dtype, device=device)

    while decoder_input.size(1) < max_len:
        # causal mask: (L, L) bool, Note: it is different from Manual version!
        decoder_mask = causal_mask(decoder_input.size(1), device)

        # calculate output
        out = model.decode(encoder_output, src_pad_mask, decoder_input, decoder_mask)

        logits = model.project(out[:, -1])        # get next token: raw scores (1, vocab_size), not probabilities
        next_word = logits.argmax(dim=-1, keepdim=True)   # get one by greedy search: (1, 1), int64, already on device

        # we add new token to the end of decoder input
        decoder_input = torch.cat([decoder_input, next_word], dim=1)

        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)   # remove the batch dimension
