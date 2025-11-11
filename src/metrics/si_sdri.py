import math
from copy import deepcopy

import torch

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import sdri_func

MAX_SI_SDR = 50


class SiSdri(BaseMetric):
    def __init__(self, compare="first", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert compare in [
            "first",
            "second",
            "average",
        ], "You should compare only the first and the second wavs pred and initial"
        self.compare = compare

    def __call__(
        self,
        audio_pred_first,
        audio_first,
        audio_pred_second,
        audio_second,
        audio_mix,
        **batch,
    ):
        sdri = []
        loss_func = sdri_func().to(audio_pred_first.device)
        for est_1, est_2, target_1, target_2, mix in zip(
            audio_pred_first, audio_pred_second, audio_first, audio_second, audio_mix
        ):
            loss1 = (
                loss_func(est_1, target_1, mix) + loss_func(est_2, target_2, mix)
            ) / 2
            loss2 = (
                loss_func(est_1, target_2, mix) + loss_func(est_2, target_1, mix)
            ) / 2
            if loss1 > loss2:
                sdri.append(loss1)
            else:
                sdri.append(loss2)
        return sum(sdri) / len(sdri)
