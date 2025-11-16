import torch
from sru import SRU

import torch.nn as nn

from src.model.rtfs_parts.layers import CompressorBlock, ReconstructionBlock, TFSelfAttention


class RTFSBlock(nn.Module):
    """
    RTFS Block (simplified implementation of paper description).
    
    Steps:
      1. Compress channels C_a -> D via 1x1 conv.
      2. Multi-scale feature generation (adaptive pooling).
      3. Dual-path processing: frequency then time with SRU layers.
      4. Self-attention over TF plane.
      5. TF-AR reconstruction units (multi-scale upsampling with U-Net style skips).
      6. Project back to C_a.

    See https://arxiv.org/abs/2309.17189 for reference.
    """
    def __init__(
        self,
        in_channels: int,
        compressed_channels: int,
        num_scales: int = 3,
        downsample_units: int = 2,
        sru_hidden_size: int = 128,
        heads: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_scales = num_scales
        self.D = compressed_channels

        self.compressor = CompressorBlock(
            in_channels=in_channels,
            out_channels=compressed_channels,
            downsample_units=downsample_units,
            downsample_factor=2,
            dimensionality_type='2D',
        )
        self.reconstructor = ReconstructionBlock(
            in_channels=compressed_channels,
            out_channels=in_channels,
            upsample_units=downsample_units,
            dimensionality_type='2D',
        )

        # SRU stacks for dual-path on compressed channels D
        self.freq_pathway = nn.Sequential(
            nn.Unfold(kernel_size=(1, 8)),
            SRU(self.D * 8, sru_hidden_size, num_layers=1, bidirectional=True),
            # Transpose convolution to restore shape
            nn.ConvTranspose2d(self.D * 8, self.D, kernel_size=(1, 8)),
        )
        self.time_pathway = nn.Sequential(
            nn.Unfold(kernel_size=(1, 8)),
            SRU(self.D * 8, sru_hidden_size, num_layers=1, bidirectional=True),
            # Transpose convolution to restore shape
            nn.ConvTranspose2d(self.D * 8, self.D, kernel_size=(1, 8)),
        )

        # Full-band self-attention block
        self.attn = TFSelfAttention(channels=self.D, num_heads=heads)

    def _dual_path(self, ag: torch.Tensor) -> torch.Tensor:
        """
        ag: (B, D, T', F') compressed multi-scale aggregate.
        Return restored (B,D,T,F).
        """
        # Frequency pathway
        B, D, T, F = ag.shape
        freq_out = self.freq_pathway(ag)  # (B, D, T, F)
        freq_out = freq_out + ag  # Residual

        freq_out = freq_out.transpose(1, 2).contiguous()

        # Time pathway
        time_out = self.time_pathway(ag)  # (B, D, T, F)
        time_out = time_out + freq_out  # Residual

        return time_out 

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """
        A: (B, C_a, T, F)
        Returns: (B, C_a, T, F)
        """
        # Compress and multi-scale feature generation
        xs = self.compressor(A)
        x = xs[-1]  # (B,D,T',F')

        # Dual-path + attention
        dp = self._dual_path(x, original_tf=A.shape[-2:])
        # TF-domain self-attention over the dual-path output
        dp = self.attn(dp) + dp

        # TF-AR reconstruction
        out = self.reconstructor(xs, dp) # (B,C_a,T,F)

        return out
