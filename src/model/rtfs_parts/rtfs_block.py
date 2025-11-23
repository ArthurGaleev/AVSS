import torch
import torch.nn as nn

from src.model.rtfs_parts.layers import (
    CompressorBlock,
    ReconstructionBlock,
    TFSelfAttention,
)


class RTFSBlock(nn.Module):
    """
    RTFS Block: Time-Frequency feature processor with dual-path architecture.

    Implements the core processing block of RTFS-Net with:
    1. Channel compression C_a -> D via 1x1 convolution
    2. Multi-scale feature generation using adaptive pooling
    3. Dual-path processing: frequency pathway followed by time pathway (using SRU)
    4. Self-attention across the time-frequency plane
    5. Multi-scale reconstruction with U-Net style skip connections (TF-AR units)
    6. Channel restoration D -> C_a via 1x1 convolution

    Reference: https://arxiv.org/abs/2309.17189
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
        """
        Initialize RTFS Block.

        Args:
            in_channels: Input channel dimension (C_a).
            compressed_channels: Channel dimension after compression (D).
            downsample_units: Number of downsampling scales in compressor (default: 2).
            unfold_kernel_size: Kernel size for unfolding in dual-path (default: 8).
            sru_hidden_size: Hidden state dimension for SRU layers (default: 128).
            sru_num_layers: Number of SRU layers (default: 4).
            attention_heads: Number of multi-head attention heads (default: 4).
            freqencies: Number of frequency bins before compression (F) (default: 257).
        """
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
            dimensionality_type="2D",
        )
        self.reconstructor = ReconstructionBlock(
            in_channels=compressed_channels,
            out_channels=in_channels,
            upsample_units=downsample_units,
            dimensionality_type="2D",
        )

        # SRU stacks for dual-path on compressed channels D
        self.unfold = nn.Unfold(kernel_size=(unfold_kernel_size, 1))

        self.freq_sru = nn.LSTM(
            self.D * unfold_kernel_size,
            sru_hidden_size,
            num_layers=sru_num_layers,
            bidirectional=True,
        )
        self.time_sru = nn.LSTM(
            self.D * unfold_kernel_size,
            sru_hidden_size,
            num_layers=sru_num_layers,
            bidirectional=True,
        )

        self.freq_tconv = nn.ConvTranspose1d(
            2 * sru_hidden_size, self.D, kernel_size=unfold_kernel_size
        )
        self.time_tconv = nn.ConvTranspose1d(
            2 * sru_hidden_size, self.D, kernel_size=unfold_kernel_size
        )

        # Full-band self-attention block
        self.attn = TFSelfAttention(
            channels=self.D,
            freqencies=freqencies // (2**downsample_units),
            num_heads=attention_heads,
        )

    def _dual_path(self, aggrigated: torch.Tensor) -> torch.Tensor:
        """
        Dual-path feature processing: frequency pathway followed by time pathway.

        Args:
            aggrigated: Compressed multi-scale aggregate of shape (B, D, T_down, F_down).

        Returns:
            Processed features of shape (B, D, T_down, F_down) with residual connections.
        """
        B, D, T, F = aggrigated.shape

        # Frequency pathway - process along frequency dimension
        x = aggrigated.transpose(1, 2).contiguous().view(-1, D, F, 1)
        x = self.unfold(x)
        x = x.permute(2, 0, 1).contiguous()
        x, _ = self.freq_sru(x)
        x = x.permute(1, 2, 0).contiguous()
        x = self.freq_tconv(x)
        freq_out = x.view(B, T, D, F).transpose(1, 2)
        freq_out = freq_out + aggrigated

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
        Process audio features through RTFS block.

        Args:
            A: Audio features of shape (B, C_a, T, F).

        Returns:
            Processed features of shape (B, C_a, T, F).
        """
        xs, x = self.compressor(A)

        dp = self._dual_path(x)
        dp = self.attn(dp) + dp

        out = self.reconstructor(xs, dp)

        return out
