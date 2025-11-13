import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.sdr import SignalDistortionRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility


def _project(u, v):
    """project vector u onto vector v."""
    scalar = (u * v).sum(dim=-1, keepdim=True) / v.pow(2).sum(dim=-1, keepdim=True)
    return scalar * v


# def sdr(est, target):
#     e_infer = _project(est, target)
#     diff = est - target - e_infer
#     ratio = (est ** 2).sum(dim=-1) / (diff ** 2).sum(dim=-1)
#     return 10 * torch.log10(ratio)


def sdr(est, target):
    return SignalDistortionRatio()(est, target)


def snr(est, target):
    diff = est - target
    ratio = target.pow(2).sum(dim=-1) / diff.pow(2).sum(dim=-1)
    return 10 * torch.log10(ratio)


def si_sdr(est, target):
    return snr(est, _project(est, target))


def si_snr(est, target):
    target = target - target.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)

    return snr(est, _project(est, target))


def si_sdr_i(est, target, mixture):
    return si_sdr(est, target) - si_sdr(mixture, target)


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
