from typing import Tuple

import torch
from torch import nn

from src.transforms.stft_transfrom import TransformSTFT


class RTFSDecoder(nn.Module):
    """STFT-based RTFS decoder as described in the paper.

    The decoder takes separated audio features in the TF domain and
    reconstructs the estimated waveform via a transposed 2D
    convolution followed by iSTFT.

    Expected input shape
    --------------------
    `z` is assumed to be of shape (B, C_z, T, F) (time, then frequency).
    Internally, we work with (B, C_z, T, F) and use a 2D
    transposed convolution with a 3x3 kernel and 2 output
    channels, corresponding to real and imaginary parts of the STFT.
    """

    def __init__(
        self,
        in_channels: int,
    ):
        super().__init__()
        self.in_channels = in_channels

        # 2D transposed convolution to generate real and imag parts
        self.deconv2d = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=2,
            kernel_size=3,
            padding=1,
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode separated TF features into waveform.

        Args:
                z: separated audio features; shape (B, C_a, T_a, F)

        Returns:
                tuple of (stft_audio_magnitude, stft_audio_phase), each of shape (B, T, F)
        """
        # 2D deconvolution to obtain real/imag STFT components
        stft_ri = self.deconv2d(z)  # (B, 2, T, F)
        real = stft_ri[:, 0, :, :]
        imag = stft_ri[:, 1, :, :]

        stft_audio_magnitude = torch.sqrt(real**2 + imag**2)
        stft_audio_phase = torch.atan2(imag, real)

        return stft_audio_magnitude, stft_audio_phase
