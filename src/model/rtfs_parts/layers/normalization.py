import torch.nn as nn


class GlobalLayerNorm(nn.GroupNorm):
    """
    Global Layer Normalization (gLN) over (C, T, F) dimensions.
    Implemented as GroupNorm with num_groups=1.
    """
    def __init__(self, num_channels: int, eps: float = 1e-8):
        super().__init__(1, num_channels, eps=eps)

class ChannelFrequencyLayerNorm(nn.LayerNorm):
    """
    Channel-Frequency Layer Normalization (cfLN).
    Normalizes over (C, F) dimensions for each time frame.
    """
    def __init__(self, num_channels: int, num_freqs: int, eps: float = 1e-8):
        super().__init__((num_channels, num_freqs), eps=eps)
