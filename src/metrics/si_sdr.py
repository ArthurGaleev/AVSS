import math

import torch

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import sdr_func

MAX_SI_SDR = 50


class SiSdr(BaseMetric):
    def __init__(self, compare="first", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert compare in [
            "first",
            "second",
            "average",
        ], "You should compare only the first and the second wavs pred and initial"
        self.compare = compare

    def __call__(
        self, audio_pred_first, audio_first, audio_pred_second, audio_second, **batch
    ):
        audio_pred_first = [
            audio_pred_first[i, ...] for i in range(audio_pred_first.shape[0])
        ]
        audio_pred_second = [
            audio_pred_second[i, ...] for i in range(audio_pred_second.shape[0])
        ]
        sisdrs = []
        if self.compare == "first":
            ests, targets = audio_pred_first, audio_first
        elif self.compare == "second":
            ests, targets = audio_pred_second, audio_second
        else:
            ests, targets = (
                audio_pred_first + audio_pred_second,
                audio_first + audio_second,
            )
        for est, target in zip(ests, targets):
            sisdrs.append(sdr_func().to(est.device)(est, target))
        return sum(sisdrs) / len(sisdrs)
