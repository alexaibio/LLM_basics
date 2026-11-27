import torch
from tokenizers import Tokenizer

from AttentionAllYouNeed.a_model_manual.masks import causal_mask
from AttentionAllYouNeed.a_model_manual.model import Transformer


def greedy_decode(
        model: Transformer,
        source: torch.Tensor,           # (1, seq_len)
        src_attn_mask: torch.Tensor,    # (1, 1, seq_len)
        tokenizer_tgt: Tokenizer,
        max_len: int,
        device: torch.device,
        verbose: bool = False,
) -> torch.Tensor:                      # (generated_len,)
    """
    Greedy decoding - simplest text generation, the next token is the one with highest probability
        - Encoder - process the input sentence ONCE, then form adjusted token embeddings.
        - Decoder - use encoded sequence (without recalculation) to generate toke, by token till EOS or max_length.
    Alternatives are: Beam Search, Random Sampling.
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Run encoder once and reuse it for every step
    encoder_output = model.encode(source, src_attn_mask)

    # Start the decoder input with the SOS token ( 1, 1) batch x first token
    decoder_input = torch.tensor([[sos_idx]], dtype=source.dtype, device=device)

    while decoder_input.size(1) < max_len:
        # build causal mask for target: we dont want decoder to see future tokens
        decoder_mask = causal_mask(decoder_input.size(1), device)

        # calculate output
        out = model.decode(encoder_output, src_attn_mask, decoder_input, decoder_mask)

        logits = model.project(out[:, -1])        # get next token: raw scores (1, vocab_size), not probabilities
        next_word = logits.argmax(dim=-1, keepdim=True)   # get one by greedy search: (1, 1), int64, already on device

        # we add new token to the end of decoder input
        decoder_input = torch.cat([decoder_input, next_word], dim=1)

        # print each word as it is generated (demo); off during validation
        if verbose:
            print(tokenizer_tgt.decode([next_word.item()]), end=' ')

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)   # wqueeze to remove the batch dimension


def beam_search_decode(
        model: Transformer,
        source: torch.Tensor,           # (1, seq_len)
        source_mask: torch.Tensor,      # (1, 1, seq_len)
        tokenizer_tgt: Tokenizer,
        max_len: int,
        device: torch.device,
        beam_size: int = 3,
) -> torch.Tensor:                      # (generated_len,)
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_initial_input = torch.tensor([[sos_idx]], dtype=source.dtype, device=device)   # (1, 1)

    # Create a candidate list: (tokens, score), score = sum of log-probabilities, log(1) = 0 at start
    candidates = [(decoder_initial_input, 0.0)]

    while True:

        # If a candidate has reached the maximum length, it means we have run the decoding for at least max_len iterations, so stop the search
        if any([cand.size(1) == max_len for cand, _ in candidates]):
            break

        # Create a new list of candidates
        new_candidates = []

        for candidate, score in candidates:

            # Do not expand candidates that have reached the eos token, but keep them in the beam
            if candidate[0][-1].item() == eos_idx:
                new_candidates.append((candidate, score))
                continue

            # Build the candidate's mask
            candidate_mask = causal_mask(candidate.size(1), device)
            # calculate output
            out = model.decode(encoder_output, source_mask, candidate, candidate_mask)
            # get next token log-probabilities: logits are raw scores, log_softmax normalizes them per step
            logits = model.project(out[:, -1])                  # (1, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)       # (1, vocab_size), log P(token | prefix) <= 0
            # get the top k candidates
            topk_log_prob, topk_idx = torch.topk(log_probs, beam_size, dim=1)
            for i in range(beam_size):
                # for each of the top k candidates, get the token and its log-probability
                token = topk_idx[:, i:i+1]                      # (1, 1), slicing keeps both dimensions
                token_log_prob = topk_log_prob[0][i].item()
                # create a new candidate by appending the token to the current candidate
                new_candidate = torch.cat([candidate, token], dim=1)
                # sum of log-probabilities = log of the product of probabilities of the whole sequence
                new_candidates.append((new_candidate, score + token_log_prob))

        # Sort the new candidates by their score
        candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
        # Keep only the top k candidates
        candidates = candidates[:beam_size]

        # If all the candidates have reached the eos token, stop
        if all([cand[0][-1].item() == eos_idx for cand, _ in candidates]):
            break

    # Return the best candidate
    return candidates[0][0].squeeze(0)   # remove the batch dimension, as in greedy_decode