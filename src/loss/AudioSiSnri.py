from typing import Dict

import torch
from torch import Tensor

from src.metrics.utils import si_snri_func


class AudioSiSnri(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        audio_first,
        audio_second,
        audio_pred_first,
        audio_pred_second,
        audio_mix,
        **batch,
    ) -> Dict[str, Tensor]:
        mse_loss_1 = (
            -(
                si_snri_func().to(audio_pred_first.device)(
                    audio_pred_first, audio_first, audio_mix
                )
                + si_snri_func().to(audio_pred_first.device)(
                    audio_pred_second, audio_second, audio_mix
                )
            )
            / 2
        )
        mse_loss_2 = (
            -(
                si_snri_func().to(audio_pred_first.device)(
                    audio_pred_second, audio_first, audio_mix
                )
                + si_snri_func().to(audio_pred_first.device)(
                    audio_pred_first, audio_second, audio_mix
                )
            )
            / 2
        )
        return {"loss": mse_loss_1 if (mse_loss_1 < mse_loss_2) else mse_loss_2}
