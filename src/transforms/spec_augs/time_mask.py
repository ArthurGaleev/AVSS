from typing import Any

import torch
import torch_audiomentations
from torch import Tensor, nn


class TimeMask(nn.Module):
    """Time masking augmentation for spectrograms.

    Args:
        ps: maximum proportion of columns to mask per mask
        num_masks: number of masks to apply
    """

    def __init__(self, ps: float = 0.05, num_masks: int = 10, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ps = ps
        self.num_masks = num_masks

    def __call__(self, data: Tensor) -> Tensor:
        spectrogram = data.clone()
        for _ in range(self.num_masks):
            ps_current = torch.rand(1).item() * self.ps
            n_columns = max(int(ps_current * spectrogram.shape[2]), 1)
            spectrogram = self.mask_random_columns(spectrogram, n_columns)
        return spectrogram

    def mask_random_columns(self, spectrogram, n_colums):
        start = torch.randint(0, spectrogram.shape[2] - n_colums, (1,)).item()
        spectrogram[:, :, start : start + n_colums] = 0
        return spectrogram
