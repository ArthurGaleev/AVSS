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
        self.unfold = nn.Unfold(kernel_size=(unfold_kernel_size, 1))
        
        self.freq_sru = SRU(self.D * unfold_kernel_size, sru_hidden_size, num_layers=sru_num_layers, bidirectional=True)
        self.time_sru = SRU(self.D * unfold_kernel_size, sru_hidden_size, num_layers=sru_num_layers, bidirectional=True)
        
        self.freq_tconv = nn.ConvTranspose1d(2 * sru_hidden_size, self.D, kernel_size=unfold_kernel_size)
        self.time_tconv = nn.ConvTranspose1d(2 * sru_hidden_size, self.D, kernel_size=unfold_kernel_size)

        # Full-band self-attention block
        self.attn = TFSelfAttention(channels=self.D, freqencies=freqencies // (2 ** downsample_units), num_heads=attention_heads)

    def _dual_path(self, ag: torch.Tensor) -> torch.Tensor:
        """
        ag: (B, D, T', F') compressed multi-scale aggregate.
        Return restored (B,D,T',F').
        """
        B, D, T, F = ag.shape
        
        # Frequency pathway - process along frequency dimension
        x = ag.transpose(1, 2).contiguous().view(-1, D, F, 1)
        x = self.unfold(x)
        x = x.permute(2, 0, 1).contiguous()
        x, _ = self.freq_sru(x)
        x = x.permute(1, 2, 0).contiguous()
        x = self.freq_tconv(x)
        freq_out = x.view(B, T, D, F).transpose(1, 2)
        freq_out = freq_out + ag

        # Time pathway - process along time dimension
        x = freq_out.permute(0, 3, 1, 2).contiguous().view(-1, D, T, 1)
        x = self.unfold(x)
        x = x.permute(2, 0, 1).contiguous()
        x, _ = self.time_sru(x)
        x = x.permute(1, 2, 0).contiguous()
        x = self.time_tconv(x)
        time_out = x.view(B, F, D, T).permute(0, 2, 3, 1)
        time_out = time_out + freq_out

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
