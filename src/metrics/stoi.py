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

        stois = []
        loss_func = stoi_func(self.sample_rate).to(audio_first.device)
        for est_1, est_2, target_1, target_2 in zip(
            audio_pred_first, audio_pred_second, audio_first, audio_second
        ):
            loss1 = (loss_func(est_1, target_1) + loss_func(est_2, target_2)) / 2
            loss2 = (loss_func(est_1, target_2) + loss_func(est_2, target_1)) / 2
            if loss1 > loss2:
                stois.append(loss1)
            else:
                stois.append(loss2)
        return sum(stois) / len(stois)
