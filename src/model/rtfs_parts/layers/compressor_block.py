from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as functional
    

class CompressorBlock(nn.Module):
    """
    Simple compressor block using depthwise separable convolutions.
    """
    def __init__(self, in_channels: int, out_channels: int, downsample_units: int = 2, downsample_factor: int = 2, dimensionality_type: Literal['2D', '1D'] = '2D'):
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

        self.dimension_compression_layer = conv_class(in_channels, out_channels, kernel_size=1) # Compress to shape (B,D,T,F), where D < C_a
        
        compression_layers = []
        for _ in range(downsample_units):
            compression_layers.append(
                conv_class(out_channels, out_channels, kernel_size=4, stride=downsample_factor, padding=1, groups=out_channels)
            ) # Depthwise conv for local feature extraction

        self.tf_compression_layers = nn.Sequential(*compression_layers)

    def forward(self, x: torch.Tensor) -> tuple[List[torch.Tensor], torch.Tensor]:
        """
        x: (B, C_a, T, F) or (B, C_a, T) for 1D
        Returns: Tuple[List of multi-scale features [(B, D, T_i, F_i), ...] or [(B, D, T_i), ...], Aggregated feature (B, D, T_q, F_q) or (B, D, T_q)]
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
