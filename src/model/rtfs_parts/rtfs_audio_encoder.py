import torch
from torch import nn

class RTFSAudioEncoder(nn.Module):
    """
    RTFS Audio Encoder: converts STFT to learned auditory embeddings.

    Encodes magnitude and phase components into learnable audio representation.
    Implements E_a = Conv2D(Re(alpha) || Im(alpha)), where:
    - Re(alpha) = |alpha| * cos(phi)
    - Im(alpha) = |alpha| * sin(phi)
    """

    def __init__(self, out_channels: int, kernel_size: int = 3, padding: int = 1, bias: bool = True):
        """
        Initialize audio encoder.

        Args:
            out_channels: Output channel dimension (C_a).
            kernel_size: Kernel size for 2D convolution (default: 3).
            padding: Padding for convolution (default: 1).
            bias: Whether to use bias in convolution (default: True).
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(
        self,
        stft_audio_magnitude: torch.Tensor,
        stft_audio_phase: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode STFT magnitude and phase into learned audio representation.

        Args:
            stft_audio_magnitude: STFT magnitude of shape (B, T, F).
            stft_audio_phase: STFT phase of shape (B, T, F).

        Returns:
            Auditory embedding of shape (B, C_a, T, F).
        """
        if stft_audio_magnitude.dim() == 4:
            stft_audio_magnitude = stft_audio_magnitude.squeeze(1)
        if stft_audio_phase.dim() == 4:
            stft_audio_phase = stft_audio_phase.squeeze(1)
            
        real = stft_audio_magnitude * torch.cos(stft_audio_phase)
        imag = stft_audio_magnitude * torch.sin(stft_audio_phase)

        x = torch.stack((real, imag), dim=1)

        a0 = self.conv(x)

        return a0
