import torch
from torch import nn

class RTFSAudioEncoder(nn.Module):
    """
    Audio Encoder for RTFS model.
    
    Converts STFT (magnitude + phase) into auditory embedding.

    Implements a0 = Ea(Re(alpha) || Im(alpha)),
    where Re(alpha) = |alpha| * cos(phi), Im(alpha) = |alpha| * sin(phi).
    """

    def __init__(self, out_channels: int, kernel_size: int = 3, padding: int = 1, bias: bool = True):
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
        Forward pass of the audio encoder.
        Args:
            stft_audio_magnitude (torch.Tensor): STFT magnitude of shape (B, T, F) or (B, 1, T, F).
            stft_audio_phase (torch.Tensor): STFT phase of shape (B, T, F) or (B, 1, T, F).
        Returns:
            torch.Tensor: Auditory embedding of shape (B, out_channels, T, F).
        """

        # Expect (B, T, F) or (B, 1, T, F). Normalize to (B, T, F).
        if stft_audio_magnitude.dim() == 4:
            stft_audio_magnitude = stft_audio_magnitude.squeeze(1)
        if stft_audio_phase.dim() == 4:
            stft_audio_phase = stft_audio_phase.squeeze(1)

        # Reconstruct real and imaginary parts.
        real = stft_audio_magnitude * torch.cos(stft_audio_phase)
        imag = stft_audio_magnitude * torch.sin(stft_audio_phase)

        # Concatenate along channel axis -> (B, 2, T, F).
        x = torch.stack((real, imag), dim=1)

        # 2D convolution over (T, F).
        a0 = self.conv(x)  # (B, out_channels, T, F)
        return a0
