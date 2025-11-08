from src.metrics.base_metric import BaseMetric
from src.metrics.utils import stoi_func


class Stoi(BaseMetric):
    def __init__(self, sample_rate, compare="first", *args, **kwargs):
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
        self.sample_rate = sample_rate
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
        pesqs = []
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
            pesqs.append(stoi_func(self.sample_rate).to(est.device)(est, target).item())
        return sum(pesqs) / len(pesqs)
