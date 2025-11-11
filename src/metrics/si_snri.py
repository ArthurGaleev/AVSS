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
        si_snri = []
        loss_func = si_snri_func().to(audio_first.device)
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
                si_snri.append(loss1)
            else:
                si_snri.append(loss2)
        return sum(si_snri) / len(si_snri)
