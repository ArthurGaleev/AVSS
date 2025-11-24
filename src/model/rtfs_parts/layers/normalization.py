import torch
import torch.nn as nn


class GlobalLayerNorm(nn.GroupNorm):
    """
    Global Layer Normalization (gLN).

    Normalizes across all channel and spatial dimensions independently for each batch.
    Implemented as GroupNorm with single group (num_groups=1).
    """
    def __init__(self, num_channels: int, eps: float = 1e-8):
        super().__init__(1, num_channels, eps=eps)

class ChannelFrequencyLayerNorm(nn.LayerNorm):
    """
    Channel-Frequency Layer Normalization (cfLN).

    Normalizes over channel and frequency dimensions independently for each time frame.
    Applied as LayerNorm on (C, F) after reshaping from (B, C, T, F).
    """
    def __init__(self, num_channels: int, num_freqs: int, eps: float = 1e-8):
        super().__init__((num_channels, num_freqs), eps=eps)
        self.num_channels = num_channels
        self.num_freqs = num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel-frequency normalization.

        Args:
            x: Features of shape (B, C, T, F).

        Returns:
            Normalized features of shape (B, C, T, F).
        """
        B, C, T, F = x.shape

        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(B * T, C, F)
        x_norm = super().forward(x_reshaped)

        x_norm = x_norm.view(B, T, C, F).permute(0, 2, 1, 3).contiguous()
        return x_norm
