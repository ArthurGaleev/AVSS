import torch

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import stoi


class Stoi(BaseMetric):
    def __init__(
        self, sample_rate, compare="first", use_pit: bool = True, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert compare in [
            "first",
            "second",
            "average",
        ], "You should compare only the first and the second wavs pred and initial"
        assert sample_rate in [
            16000,
            8000,
        ], "Pesq metric is not implemented for sample rates not 16kHz or 8kHz"
        self.compare = compare
        self.metric_fn = stoi(sample_rate)
        self.use_pit = use_pit

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
                loss1 += self.metric_fn(audio_pred_first, audio_first)
                loss2 += self.metric_fn(audio_pred_second, audio_first)
            if self.compare in ["second", "average"]:
                loss1 += self.metric_fn(audio_pred_second, audio_second)
                loss2 += self.metric_fn(audio_pred_first, audio_second)
            result = torch.max(loss1, loss2).mean()
        else:
            if self.compare in ["first", "average"]:
                loss1 += self.metric_fn(audio_pred_first, audio_first)
            if self.compare in ["second", "average"]:
                loss1 += self.metric_fn(audio_pred_second, audio_second)
            result = loss1.mean()
        norm_coeff = 2 if self.compare == "average" else 1
        return result / norm_coeff
