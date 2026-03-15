import torch

def make_mask(lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
    """
    lengths: (B,)
    returns: (B, max_len) boolean mask, True for valid positions
    """
    time_idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, T)
    return time_idx < lengths.unsqueeze(1)  # (B, T)