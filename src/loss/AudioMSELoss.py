from typing import Dict

import torch
from torch import Tensor


class AudioMSELoss(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        audio_first,
        audio_second,
        audio_pred_first,
        audio_pred_second,
        **batch,
    ) -> Dict[str, Tensor]:
        mse_loss_1 = torch.nn.MSELoss()(
            audio_pred_first, audio_first
        ) + torch.nn.MSELoss()(audio_pred_second, audio_second)
        mse_loss_2 = torch.nn.MSELoss()(
            audio_pred_second, audio_first
        ) + torch.nn.MSELoss()(audio_pred_first, audio_second)
        return {"loss": mse_loss_1.requires_grad_(True) if (mse_loss_1 < mse_loss_2) else mse_loss_2.requires_grad_(True)}
