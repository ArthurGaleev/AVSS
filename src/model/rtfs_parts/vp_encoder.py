import torch

import torch.nn as nn

from src.model.rtfs_parts.layers import CompressorBlock, ReconstructionBlock, TDANetSelfAttention


class VPEncoder(nn.Module):
    """
    VP Encoder (implementation of paper description).

    See https://arxiv.org/abs/2309.17189 for reference.
    """
    def __init__(
        self,
        in_channels: int,
        compressed_channels: int,
        downsample_units: int = 2,
        heads: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.D = compressed_channels

        self.compressor = CompressorBlock(
            in_channels=in_channels,
            out_channels=compressed_channels,
            downsample_units=downsample_units,
            downsample_factor=2,
            dimensionality_type='1D',
        )
        self.reconstructor = ReconstructionBlock(
            in_channels=compressed_channels,
            out_channels=in_channels,
            upsample_units=downsample_units,
            dimensionality_type='1D',
        )

        # Full-band self-attention block
        self.attn = TDANetSelfAttention(channels=self.D, num_heads=heads)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """
        A: (B, C_a, T, F)
        Returns: (B, C_a, T, F)
        """
        # Compress and multi-scale feature generation
        xs, x = self.compressor(A)

        # TDANet self-attention
        x = self.attn(x) + x

        # TF-AR reconstruction
        out = self.reconstructor(xs, x) # (B,C_a,T,F)
        return out
