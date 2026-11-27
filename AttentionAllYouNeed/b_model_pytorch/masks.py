import torch

# Masks in the PyTorch convention: True = blocked, False = allowed
# (INVERTED compared to a_model_manual/masks.py, where 1 = allowed)


def padding_mask(tokens: torch.Tensor, pad_idx: int) -> torch.Tensor:
    # (B, S) token ids --> (B, S) bool, True = [PAD] (attention ignores these keys)
    return tokens == pad_idx


def causal_mask(size: int, device) -> torch.Tensor:
    # (L, L) bool, True above the diagonal = future positions (blocked)
    # same as manual causal_mask, but inverted: there 1 = allowed
    # built-in alternative: nn.Transformer.generate_square_subsequent_mask(size) - float, -inf = blocked
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
