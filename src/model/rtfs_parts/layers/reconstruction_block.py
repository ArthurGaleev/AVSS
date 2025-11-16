from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as functional

from src.model.rtfs_parts.layers import GlobalLayerNorm


class TFARUnit(nn.Module):
    """
    Temporal-Frequency Attention Reconstruction (simplified).
    Implements I(m,n) combining two context inputs via learned gates.
    """
    def __init__(self, channels_m: int, channels_n: int):
        super().__init__()
        self.w1_pathway = nn.Sequential(
            nn.Conv2d(channels_n, channels_n, kernel_size=4, stride=2, padding=1, groups=channels_n), # depthwise conv (same size)
            GlobalLayerNorm(channels_n), # gLN
            nn.Sigmoid(),
        )
        self.w2_pathway = nn.Sequential(
            nn.Conv2d(channels_m, channels_m, kernel_size=4, stride=2, padding=1, groups=channels_m), # depthwise conv (same size)
            GlobalLayerNorm(channels_m), # gLN
        )
        self.w3_pathway = nn.Sequential(
            nn.Conv2d(channels_n, channels_n, kernel_size=3, padding=1, groups=channels_n), # depthwise conv (same size)
            GlobalLayerNorm(channels_n), # gLN
        )

    def forward(self, m: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """
        m: (B, D, T_m, F_m)
        n: (B, D, T_n, F_n)
        Returns: (B, D, T_m, F_m)
        """

        w1 = self.w1_pathway(n)  # (B, D, T_n, F_n)
        w2 = self.w2_pathway(m)  # (B, D, T_m, F_m)
        w3 = self.w3_pathway(n)  # (B, D, T_n, F_n)

        _, _, T_m, F_m = m.shape

        # Interpolate w1, w3 to m's size
        w1_up = functional.interpolate(w1, size=(T_m, F_m), mode='bilinear', align_corners=False)
        w3_up = functional.interpolate(w3, size=(T_m, F_m), mode='bilinear', align_corners=False)

        out = w1_up * w2 + w3_up

        return out


class ReconstructionBlock(nn.Module):
    """
    Simple reconstruction block using depthwise separable convolutions.
    """
    def __init__(self, in_channels: int, out_channels: int, upsample_units: int = 2):
        super().__init__()
        
        self.g_fuse_layers = nn.ModuleList()
        self.residual_fuse_layers = nn.ModuleList()

        in_channels_m, in_channels_n = in_channels, in_channels // (2 ** upsample_units)

        for i in range(upsample_units):
            self.g_fuse_layers.append(TFARUnit(in_channels_m, in_channels_n))

            in_channels_m_next = in_channels_m // 2  # assuming downsample factor of 2 in compressor

            if i != 0:
                self.residual_fuse_layers.append(TFARUnit(in_channels_m, in_channels_m_next))

            in_channels_m = in_channels_m_next

        self.dimension_upsampling_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # Restore to original channels


    def forward(self, A_i: List[torch.Tensor], A_G: torch.Tensor) -> torch.Tensor:
        """
        xs: List of multi-scale features [(B, D, T_i, F_i), ...] (from compressor)
        A_G: (B, D, T_base, F_base) - base feature to reconstruct from
        Returns: (B, D, T, F)
        """
        A_i = [
            layer(A, A_G)
            for layer, A in zip(self.g_fuse_layers, A_i)
        ]
        x = A_i[-1] # Start from the coarsest scale (B,D,T_q,F_q)
        for layer, A in zip(reversed(self.residual_fuse_layers), reversed(A_i[:-1])):
            # FIXME: In original paper A''_{q-1-i} = I(A'_{q-1-i}, A'_{q-i}) + A_{q-1-i} = I(I(A_{q-1-i}, A_G), I(A_{q-i}, A_G)) + A_{q-1-i}
            # but I think this is a mistake and it should be:
            # A''_{q-1-i} = I(A'_{q-1-i} + A''_{q-i}) + A_{q-1-i}
            x = layer(A, x) + A
        x = self.dimension_upsampling_layer(x) # Restore to original channels (B, C_a, T, F)
        return x
