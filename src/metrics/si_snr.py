import torch

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import si_snr, si_snr_i


class SiSnr(BaseMetric):
    def __init__(
        self,
        compare="first",
        improved: bool = True,
        use_pit: bool = True,
        *args,
        **kwargs,
    ):
        assert compare in [
            "first",
            "second",
            "average",
        ], "You should compare only the first and the second wavs pred and initial"

        super().__init__(*args, **kwargs)

        self.compare = compare
        self.improved = improved
        self.use_pit = use_pit
        self.metric_fn = {
            True: si_snr_i,
            False: lambda est, target, _: si_snr(est, target),
        }[self.improved]

    def __call__(
        self,
        audio_pred_first,
        audio_first,
        audio_pred_second,
        audio_second,
        audio_mix,
        **batch,
    ):
        batch_size = audio_first.shape[0]
        loss1, loss2 = torch.zeros(batch_size, device=audio_first.device), torch.zeros(
            batch_size, device=audio_first.device
        )
        if self.use_pit:
            if self.compare in ["first", "average"]:
                loss1 += self.metric_fn(audio_pred_first, audio_first, audio_mix)
                loss2 += self.metric_fn(audio_pred_second, audio_first, audio_mix)
            if self.compare in ["second", "average"]:
                loss1 += self.metric_fn(audio_pred_second, audio_second, audio_mix)
                loss2 += self.metric_fn(audio_pred_first, audio_second, audio_mix)
            result = torch.max(loss1, loss2).mean()
        else:
            if self.compare in ["first", "average"]:
                loss1 += self.metric_fn(audio_pred_first, audio_first, audio_mix)
            if self.compare in ["second", "average"]:
                loss1 += self.metric_fn(audio_pred_second, audio_second, audio_mix)
            result = loss1.mean()
        norm_coeff = 2 if self.compare == "average" else 1
        return result / norm_coeff
