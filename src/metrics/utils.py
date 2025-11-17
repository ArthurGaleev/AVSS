from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.functional.audio.sdr import (
    scale_invariant_signal_distortion_ratio,
    signal_distortion_ratio,
)
from torchmetrics.functional.audio.snr import (
    scale_invariant_signal_noise_ratio,
    signal_noise_ratio,
)


def sdr(est, target):
    return signal_distortion_ratio(est, target)


def si_sdr(est, target):
    return scale_invariant_signal_distortion_ratio(est, target)


def si_sdr_i(est, target, mixture):
    return si_sdr(est, target) - si_sdr(mixture, target)


def snr(est, target):
    return signal_noise_ratio(est, target)


def si_snr(est, target):
    return scale_invariant_signal_noise_ratio(est, target)


def si_snr_i(est, target, mixture):
    return si_snr(est, target) - si_snr(mixture, target)


def pesq(sample_rate):
    assert sample_rate in [
        16000,
        8000,
    ], "Pesq metric is not implemented for sample rates not 16kHz or 8kHz"
    return PerceptualEvaluationSpeechQuality(sample_rate, "wb")


def stoi(sample_rate):
    assert sample_rate in [
        16000,
        10000,
    ], "Stoi metric is not implemented for sample rates not 16kHz or 10kHz"
    return ShortTimeObjectiveIntelligibility(sample_rate)