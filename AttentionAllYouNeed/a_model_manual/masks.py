import torch


def padding_mask(tokens, pad_id):
    '''
    Hide [PAD] tokens from attention: 1 = real token, 0 = PAD.
    (seq_len) -> (1, 1, seq_len)
    '''
    return (tokens != pad_id).unsqueeze(0).unsqueeze(0).int()


def causal_mask(size):
    '''
    This is crucial in autoregressive models where the prediction of each position should only depend on previous positions.
    - we dont need PAD tokens to be seen by self-attention mechanism
    - we dont need every word to see words coming after it in a sentence
    '''
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
