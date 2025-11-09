from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio,
    ScaleInvariantSignalNoiseRatio,
)
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility


def pesq_func(sample_rate):
    assert sample_rate in [
        16000,
        8000,
    ], "Pesq metric is not implemented for sample rates not 16kHz or 8kHz"
    return PerceptualEvaluationSpeechQuality(sample_rate, "wb")


class ScaleInvariantSignalNoiseRatioImprovement(ScaleInvariantSignalNoiseRatio):
    def forward(self, preds, target, mixture):
        si_snr_est = super().forward(preds, target)
        si_snr_mix = super().forward(mixture, target)
        return si_snr_est - si_snr_mix


def si_snri_func():
    return ScaleInvariantSignalNoiseRatioImprovement()


class ScaleInvariantSignalDistortionRatioImprovement(
    ScaleInvariantSignalDistortionRatio
):
    def forward(self, preds, target, mixture):
        si_sdr_est = super().forward(preds, target)
        si_sdr_mix = super().forward(mixture, target)
        return si_sdr_est - si_sdr_mix


def sdri_func():
    return ScaleInvariantSignalDistortionRatioImprovement()


def stoi_func(sample_rate):
    assert sample_rate in [
        16000,
        10000,
    ], "Stoi metric is not implemented for sample rates not 16kHz or 10kHz"
    return ShortTimeObjectiveIntelligibility(sample_rate)
