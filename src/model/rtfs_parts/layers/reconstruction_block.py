from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as functional

from src.model.rtfs_parts.layers.normalization import GlobalLayerNorm


class TFARUnit(nn.Module):
    """
    Temporal-Frequency Attention Reconstruction (simplified).
    Implements I(m,n) combining two context inputs via learned gates.
    """

    def __init__(
        self, in_channels: int, dimensionality_type: Literal["2D", "1D"] = "2D"
    ):
        super().__init__()

        self.in_channels = in_channels
        self.dimensionality_type = dimensionality_type

        conv_class = {
            "2D": nn.Conv2d,
            "1D": nn.Conv1d,
        }[dimensionality_type]

        self.w1_pathway = nn.Sequential(
            conv_class(
                in_channels,
                in_channels,
                kernel_size=4,
                padding="same",
                groups=in_channels,
            ),  # depthwise conv (same size)
            GlobalLayerNorm(in_channels),  # gLN
            nn.Sigmoid(),
        )
        self.w2_pathway = nn.Sequential(
            conv_class(
                in_channels,
                in_channels,
                kernel_size=4,
                padding="same",
                groups=in_channels,
            ),  # depthwise conv (same size)
            GlobalLayerNorm(in_channels),  # gLN
        )
        self.w3_pathway = nn.Sequential(
            conv_class(
                in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels
            ),  # depthwise conv (same size)
            GlobalLayerNorm(in_channels),  # gLN
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

        if self.dimensionality_type == "1D":
            _, _, T_m = w2.shape
            interpolate_shape = (T_m,)
        else:
            _, _, T_m, F_m = w2.shape
            interpolate_shape = (T_m, F_m)

        # Interpolate w1, w3 to m's size
        w1_up = functional.interpolate(
            w1, size=interpolate_shape, mode="nearest"
        )
        w3_up = functional.interpolate(
            w3, size=interpolate_shape, mode="nearest"
        )
        out = w1_up * w2 + w3_up

        return out


class ReconstructionBlock(nn.Module):
    """
    Simple reconstruction block using depthwise separable convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_units: int = 2,
        dimensionality_type: Literal["2D", "1D"] = "2D",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_units = upsample_units
        self.dimensionality_type = dimensionality_type

        self.g_fuse_layers = nn.ModuleList()
        self.residual_fuse_layers = nn.ModuleList()

        # Build TFAR units for each scale
        for _ in range(upsample_units + 1):
            # G-fuse: fuse this scale with the processed result (A_G has deepest_channels)
            self.g_fuse_layers.append(
                TFARUnit(in_channels, dimensionality_type=dimensionality_type)
            )

        for _ in range(upsample_units):
            # When reconstructing, we go from coarsest to finest
            self.residual_fuse_layers.append(
                TFARUnit(in_channels, dimensionality_type=dimensionality_type)
            )

        conv_class = {
            "2D": nn.Conv2d,
            "1D": nn.Conv1d,
        }[dimensionality_type]

        self.dimension_upsampling_layer = conv_class(
            in_channels, out_channels, kernel_size=1
        )  # Restore to original channels

    def forward(self, A_i: List[torch.Tensor], A_G: torch.Tensor) -> torch.Tensor:
        """
        A_i: List of multi-scale features [(B, D, T_0, F_0), (B, D, T_1, F_1), ..., (B, D, T_q, F_q)] (from compressor)
        A_G: (B, D, T_q, F_q) - processed feature from dual-path + attention
        Returns: (B, out_channels, T_0, F_0) - reconstructed feature at original resolution
        """
        # First, fuse all multi-scale features with A_G
        A_prime = []
        for layer, A in zip(self.g_fuse_layers, A_i):
            A_prime.append(layer(A, A_G))

        # Start reconstruction from the coarsest scale
        x = A_prime[-1]  # (B, D, T_q, F_q) after TFAR with A_G

        # Progressively upsample and fuse with finer scales
        for layer, A_finer, A_q in zip(
            self.residual_fuse_layers, A_prime[-2::-1], A_i[-2::-1]
        ):
            # NOTE: In original paper A''_{q-1-i} = I(A'_{q-1-i}, A'_{q-i}) + A_{q-1-i} = I(I(A_{q-1-i}, A_G), I(A_{q-i}, A_G)) + A_{q-1-i}
            # but I think this is a mistake and it should be:
            # A''_{q-1-i} = I(A'_{q-1-i} + A''_{q-i}) + A_{q-1-i}

            # Fuse current result with next finer scale
            x = layer(A_finer, x) + A_q

        # Final projection to restore original channel dimension
        x = self.dimension_upsampling_layer(x)  # (B, out_channels, T_0, F_0)
        return x
