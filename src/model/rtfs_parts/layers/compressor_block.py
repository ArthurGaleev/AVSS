from typing import List
import torch

import torch.nn as nn
import torch.nn.functional as functional
    

class CompressorBlock(nn.Module):
    """
    Simple compressor block using depthwise separable convolutions.
    """
    def __init__(self, in_channels: int, out_channels: int, downsample_units: int = 2, downsample_factor: int = 2):
        super().__init__()

        self.dimension_compression_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1) # Compress to shape (B,D,T,F), where D < C_a
        
        compression_layers = []
        in_channels = out_channels
        for _ in range(downsample_units):
            compression_layers.append(
                nn.Conv2d(in_channels, in_channels * downsample_factor, kernel_size=4, stride=downsample_factor, padding=1, groups=in_channels)
            ) # Depthwise conv for local feature extraction
            in_channels *= downsample_factor

        self.tf_compression_layers = nn.Sequential(*compression_layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        x: (B, C_a, T, F)
        Returns: List of multi-scale features [(B, D, T_i, F_i), ...]
        """
        x = self.dimension_compression_layer(x)
        
        xs = [x]
        for layer in self.tf_compression_layers:
            x = layer(x)
            xs.append(x)

        _, _, T_q, F_q = x.shape

        # Perform adaptive average pooling to get fixed size representation
        for x_ in xs[:-1]:
            x += functional.adaptive_avg_pool2d(x_, (T_q, F_q))

        return xs
