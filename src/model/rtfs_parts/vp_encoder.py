import torch

import torch.nn as nn

from src.model.rtfs_parts.layers import CompressorBlock, ReconstructionBlock, TDANetSelfAttention


class VPEncoder(nn.Module):
    """
    VP Encoder: processes video embeddings for audio-visual fusion.

    Encodes video features via:
    1. Multi-scale compression with adaptive pooling
    2. TDANet self-attention for global temporal modeling
    3. Multi-scale reconstruction with U-Net style skip connections

    Reference: https://arxiv.org/abs/2309.17189
    """
    def __init__(
        self,
        in_channels: int,
        compressed_channels: int,
        downsample_units: int = 2,
        heads: int = 4,
    ):
        """
        Initialize video encoder.

        Args:
            in_channels: Input channel dimension (C_v).
            compressed_channels: Channel dimension after compression (D).
            downsample_units: Number of downsampling scales (default: 2).
            heads: Number of multi-head attention heads (default: 4).
        """
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
        Process video embeddings through VP encoder.

        Args:
            A: Video embeddings of shape (B, C_v, T).

        Returns:
            Processed video features of shape (B, C_v, T).
        """
        xs, x = self.compressor(A)

        x = self.attn(x) + x

        out = self.reconstructor(xs, x)
        return out
