from src.metrics.base_metric import BaseMetric
from src.metrics.utils import pesq
import torch


class Pesq(BaseMetric):
    def __init__(self, sample_rate, compare="first", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert compare in [
            "first",
            "second",
            "average",
        ], "You should compare only the first and the second wavs pred and initial"
        self.compare = compare
        self.metric_fn = pesq(sample_rate)

    def __call__(
        self, audio_pred_first, audio_first, audio_pred_second, audio_second, **batch
    ):
        batch_size = audio_first.shape[0]
        loss1, loss2 = torch.zeros(batch_size, device=audio_first.device), torch.zeros(
            batch_size, device=audio_first.device
        )
        if self.compare in ["first", "average"]:
            loss1 += self.metric_fn(audio_pred_first, audio_first)
            loss2 += self.metric_fn(audio_pred_second, audio_first)
        if self.compare in ["second", "average"]:
            loss1 += self.metric_fn(audio_pred_second, audio_second)
            loss2 += self.metric_fn(audio_pred_first, audio_second)
        result = torch.max(loss1, loss2).mean()

        norm_coeff = 2 if self.compare == "average" else 1
        return result / norm_coeff
