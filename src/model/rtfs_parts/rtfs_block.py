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
        downsample_units: int = 2,
        unfold_kernel_size: int = 8,
        sru_hidden_size: int = 128,
        sru_num_layers: int = 4,
        attention_heads: int = 4,
        freqencies: int = 257,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.D = compressed_channels
        self.downsample_units = downsample_units
        self.unfold_kernel_size = unfold_kernel_size
        self.sru_hidden_size = sru_hidden_size
        self.sru_num_layers = sru_num_layers
        self.heads = attention_heads
        self.freqencies = freqencies

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
            nn.Unfold(kernel_size=(1, unfold_kernel_size)),
            SRU(self.D * unfold_kernel_size, sru_hidden_size, num_layers=sru_num_layers, bidirectional=True),
            # Transpose convolution to restore shape
            nn.ConvTranspose1d(self.D * unfold_kernel_size, self.D, kernel_size=unfold_kernel_size),
        )
        self.time_pathway = nn.Sequential(
            nn.Unfold(kernel_size=(1, unfold_kernel_size)),
            SRU(self.D * unfold_kernel_size, sru_hidden_size, num_layers=sru_num_layers, bidirectional=True),
            # Transpose convolution to restore shape
            nn.ConvTranspose1d(self.D * unfold_kernel_size, self.D, kernel_size=unfold_kernel_size),
        )

        # Full-band self-attention block
        self.attn = TFSelfAttention(channels=self.D, freqencies=freqencies, num_heads=attention_heads)

    def _dual_path(self, ag: torch.Tensor) -> torch.Tensor:
        """
        ag: (B, D, T', F') compressed multi-scale aggregate.
        Return restored (B,D,T',F').
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
        xs, x = self.compressor(A)

        # Dual-path + attention
        dp = self._dual_path(x)
        # TF-domain self-attention over the dual-path output
        dp = self.attn(dp) + dp

        # TF-AR reconstruction
        out = self.reconstructor(xs, dp) # (B,C_a,T,F)

        return out
