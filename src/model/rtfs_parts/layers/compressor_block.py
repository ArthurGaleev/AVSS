from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as functional
    

class CompressorBlock(nn.Module):
    """
    Multi-scale Compressor: downsamples features and generates multi-scale representation.

    Compresses channel and spatial dimensions via depthwise separable convolutions
    while maintaining multi-scale features for later reconstruction with skip connections.
    """
    def __init__(self, in_channels: int, out_channels: int, downsample_units: int = 2, downsample_factor: int = 2, dimensionality_type: Literal['2D', '1D'] = '2D'):
        """
        Initialize compressor block.

        Args:
            in_channels: Input channel dimension.
            out_channels: Output channel dimension (D).
            downsample_units: Number of downsampling scales (default: 2).
            downsample_factor: Downsampling factor per unit (default: 2).
            dimensionality_type: '2D' for (T, F) or '1D' for (T,) features (default: '2D').
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample_units = downsample_units
        self.downsample_factor = downsample_factor
        self.dimensionality_type = dimensionality_type

        conv_class = {
            '2D': nn.Conv2d,
            '1D': nn.Conv1d,
        }[dimensionality_type]

        self.dimension_compression_layer = conv_class(in_channels, out_channels, kernel_size=1)
        
        compression_layers = []
        for _ in range(downsample_units):
            compression_layers.append(
                conv_class(out_channels, out_channels, kernel_size=4, stride=downsample_factor, padding=1, groups=out_channels)
            )

        self.tf_compression_layers = nn.Sequential(*compression_layers)

    def forward(self, x: torch.Tensor) -> tuple[List[torch.Tensor], torch.Tensor]:
        """
        Compress features and generate multi-scale representation.

        Args:
            x: Input features of shape (B, C_in, T, F) for 2D or (B, C_in, T) for 1D.

        Returns:
            Tuple of (list of multi-scale features, aggregated final feature):
                - Multi-scale: list of features at each compression level
                - Aggregated: sum of all features pooled to finest resolution
        """
        x = self.dimension_compression_layer(x)
        
        xs = [x]
        for layer in self.tf_compression_layers:
            x = layer(x)
            xs.append(x)

        if self.dimensionality_type == '1D':
            _, _, T_q = x.shape
            pooling_fn = lambda x_: functional.adaptive_avg_pool1d(x_, (T_q,))
        elif self.dimensionality_type == '2D':
            _, _, T_q, F_q = x.shape
            pooling_fn = lambda x_: functional.adaptive_avg_pool2d(x_, (T_q, F_q))

        # Aggregate multi-scale features by pooling to same size and adding
        x_agg = xs[-1]
        for x_ in xs[:-1]:
            x_agg = x_agg + pooling_fn(x_)
        
        return xs, x_agg
