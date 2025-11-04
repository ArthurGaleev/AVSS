from typing import Dict

import torch
from torch import Tensor


class SpectrogramLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        spectrogram_first,
        spectrogram_second,
        spectrogram_pred_first,
        spectrogram_pred_second,
        **batch,
    ) -> Dict[str, Tensor]:
        mse_loss_1 = torch.nn.MSELoss()(
            spectrogram_pred_first, spectrogram_first
        ) + torch.nn.MSELoss()(spectrogram_pred_second, spectrogram_second)
        mse_loss_2 = torch.nn.MSELoss()(
            spectrogram_pred_second, spectrogram_first
        ) + torch.nn.MSELoss()(spectrogram_pred_first, spectrogram_second)
        return {"loss": mse_loss_1 if (mse_loss_1 < mse_loss_2) else mse_loss_2}
