from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as functional

from src.model.rtfs_parts.layers.normalization import GlobalLayerNorm


class TFARUnit(nn.Module):
    """
    Temporal-Frequency Attention Reconstruction (TFAR) Unit.

    Implements adaptive gating for multi-scale feature fusion using learned gates.
    Combines two context inputs: m (finer) and n (coarser) via learned gate functions.
    """

    def __init__(
        self, in_channels: int, dimensionality_type: Literal["2D", "1D"] = "2D"
    ):
        """
        Initialize TFAR unit.

        Args:
            in_channels: Channel dimension.
            dimensionality_type: '2D' for (T, F) or '1D' for (T,) features (default: '2D').
        """
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
            ),
            GlobalLayerNorm(in_channels),
            nn.Sigmoid(),
        )
        self.w2_pathway = nn.Sequential(
            conv_class(
                in_channels,
                in_channels,
                kernel_size=4,
                padding="same",
                groups=in_channels,
            ),
            GlobalLayerNorm(in_channels),
        )
        self.w3_pathway = nn.Sequential(
            conv_class(
                in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels
            ),
            GlobalLayerNorm(in_channels),
        )

    def forward(self, m: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """
        Apply TFAR gating to combine two features.

        Args:
            m: Finer scale features of shape (B, D, T_m, F_m) or (B, D, T_m).
            n: Coarser scale features of shape (B, D, T_n, F_n) or (B, D, T_n).

        Returns:
            Gated fusion of shape (B, D, T_m, F_m) or (B, D, T_m).
        """
        w1 = self.w1_pathway(n)
        w2 = self.w2_pathway(m)
        w3 = self.w3_pathway(n)

        if self.dimensionality_type == "1D":
            _, _, T_m = w2.shape
            interpolate_shape = (T_m,)
        else:
            _, _, T_m, F_m = w2.shape
            interpolate_shape = (T_m, F_m)

        w1_up = functional.interpolate(w1, size=interpolate_shape, mode="nearest")
        w3_up = functional.interpolate(w3, size=interpolate_shape, mode="nearest")
        out = w1_up * w2 + w3_up

        return out


class ReconstructionBlock(nn.Module):
    """
    Multi-scale Reconstructor: upsamples features and fuses multi-scale information.

    Progressively reconstructs features from coarsest to finest scale using
    TFAR units for gated fusion with skip connections from compression stages.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_units: int = 2,
        dimensionality_type: Literal["2D", "1D"] = "2D",
    ):
        """
        Initialize reconstruction block.

        Args:
            in_channels: Compressed channel dimension (D).
            out_channels: Output channel dimension (C_in).
            upsample_units: Number of upsampling scales (default: 2).
            dimensionality_type: '2D' for (T, F) or '1D' for (T,) features (default: '2D').
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_units = upsample_units
        self.dimensionality_type = dimensionality_type

        self.g_fuse_layers = nn.ModuleList()
        self.residual_fuse_layers = nn.ModuleList()

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
        )

    def forward(self, A_i: List[torch.Tensor], A_G: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct features from multi-scale compressed representation.

        Args:
            A_i: List of multi-scale features from compressor at each scale.
            A_G: Processed coarse features from dual-path + attention.

        Returns:
            Reconstructed features at original resolution of shape (B, C_out, T_0, F_0).
        """
        A_prime = []
        for layer, A in zip(self.g_fuse_layers, A_i):
            A_prime.append(layer(A, A_G))

        x = A_prime[-1]

        # Progressively upsample and fuse with finer scales
        for layer, A_finer, A_q in zip(
            self.residual_fuse_layers, A_prime[-2::-1], A_i[-2::-1]
        ):
            # NOTE: In original paper A''_{q-1-i} = I(A'_{q-1-i}, A'_{q-i}) + A_{q-1-i} = I(I(A_{q-1-i}, A_G), I(A_{q-i}, A_G)) + A_{q-1-i}
            # but I think this is a mistake and it should be:
            # A''_{q-1-i} = I(A'_{q-1-i} + A''_{q-i}) + A_{q-1-i}

            # Fuse current result with next finer scale
            x = layer(A_finer, x) + A_q

        x = self.dimension_upsampling_layer(x)
        return x
