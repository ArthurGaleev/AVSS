from typing import Tuple

import torch
from torch import nn


class RTFSDecoder(nn.Module):
    """
    RTFS Decoder: reconstructs STFT components from separated audio features.

    Converts latent time-frequency representation back to magnitude and phase
    spectrogram via 2D transposed convolution with real/imaginary decomposition.
    The magnitude and phase can be used with iSTFT for waveform reconstruction.
    """

    def __init__(
        self,
        in_channels: int,
    ):
        """
        Initialize RTFS Decoder.

        Args:
            in_channels: Channel dimension of input features (C_z).
        """
        super().__init__()

        self.in_channels = in_channels

        self.deconv2d = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=2,
            kernel_size=3,
            padding=1,
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode separated features to magnitude and phase spectrograms.

        Args:
            z: Separated audio features of shape (B, C_z, T, F).

        Returns:
            Tuple of (magnitude, phase) spectrograms, each of shape (B, T, F).
        """
        # 2D deconvolution to obtain real/imag STFT components
        stft_ri = self.deconv2d(z)
        real = stft_ri[:, 0, :, :]
        imag = stft_ri[:, 1, :, :]
        
        stft_audio_magnitude = torch.sqrt(real**2 + imag**2)
        stft_audio_phase = torch.atan2(imag, real)

        return stft_audio_magnitude, stft_audio_phase
