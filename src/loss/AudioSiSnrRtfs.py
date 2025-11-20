from typing import Dict

import torch
from torch import Tensor

from src.metrics.si_snr import SiSnr


class AudioSiSnrRTFS(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.metric = SiSnr(
            improved=False, use_pit=False
        )  # Avoid calculating constant term

    def forward(
        self,
        **batch,
    ) -> Dict[str, Tensor]:
        return {"loss": -self.metric(**batch)}
