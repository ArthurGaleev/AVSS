from typing import Any

import torch
import torch_audiomentations
from torch import Tensor, nn


class MaskFreq(nn.Module):
    """Frequency masking augmentation for spectrograms.
    Args:
        F: width of frequency mask.
    """

    def __init__(self, F: int = 27, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.F = F

    def __call__(self, data: Tensor):
        spectrogram = data.clone()
        start = torch.randint(0, spectrogram.shape[1] - self.F, (1,)).item()
        spectrogram[:, start : start + self.F, :] = 0
        return spectrogram
