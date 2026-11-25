import torch

from AttentionAllYouNeed.a_model_manual.masks import causal_mask


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device, verbose=False):
    """
    Greedy decoding - simplest text generation, the next token is the one with highest probability
        - Encoder - process the input sentence ONCE, then form adjusted token embeddings.
        - Decoder - use encoded sequence (without recalculation) to generate toke, by token till EOS or max_length.
    Alternatives are: Beam Search, Random Sampling.
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Run encoder once and reuse it for every step
    encoder_output = model.encode(source, source_mask)

    # We start the decoder input with the sos token ( 1, 1) batch x first token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # build causel mask for target: we dont want decoder to see future tokens
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(out[:, -1])        # get next token
        _, next_word = torch.max(prob, dim=1)   # get one by greedy search

        # we add new token to the end of decoder input
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
            dim=1
        )

        # print each word as it is generated (demo); off during validation
        if verbose:
            print(tokenizer_tgt.decode([next_word.item()]), end=' ')

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)   # wqueeze to remove the batch dimension
