from copy import deepcopy

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import si_snri_func


class SiSnri(BaseMetric):
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
        audio_pred_first = [
            audio_pred_first[i, ...] for i in range(audio_pred_first.shape[0])
        ]
        audio_pred_second = [
            audio_pred_second[i, ...] for i in range(audio_pred_second.shape[0])
        ]
        audio_mix = [audio_mix[i, ...] for i in range(audio_mix.shape[0])]
        si_snri = []
        if self.compare == "first":
            ests, targets, mixes = audio_pred_first, audio_first, audio_mix
        elif self.compare == "second":
            ests, targets, mixes = audio_pred_second, audio_second, audio_mix
        else:
            ests, targets, mixes = (
                audio_pred_first + audio_pred_second,
                audio_first + audio_second,
                deepcopy(audio_mix) + deepcopy(audio_mix),
            )
        for est, target, mix in zip(ests, targets, mixes):
            si_snri.append(si_snri_func().to(est.device)(est, target, mix))
        return sum(si_snri) / len(si_snri)
