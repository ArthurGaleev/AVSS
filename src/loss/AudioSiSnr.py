from typing import Dict

import torch
from torch import Tensor

from src.metrics.utils import si_snr_func


class AudioSiSnr(torch.nn.Module):
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
        mse_loss_1 = si_snr_func(audio_pred_first, audio_first) + si_snr_func(
            audio_pred_second, audio_second
        )
        mse_loss_2 = si_snr_func(audio_pred_second, audio_first) + si_snr_func(
            audio_pred_first, audio_second
        )
        return {"loss": mse_loss_1 if (mse_loss_1 < mse_loss_2) else mse_loss_2}
