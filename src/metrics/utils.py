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


def si_snr_func():
    return ScaleInvariantSignalNoiseRatio()


def sdr_func():
    return ScaleInvariantSignalDistortionRatio()


def stoi_func(sample_rate):
    assert sample_rate in [
        16000,
        10000,
    ], "Stoi metric is not implemented for sample rates not 16kHz or 10kHz"
    return ShortTimeObjectiveIntelligibility(sample_rate)
