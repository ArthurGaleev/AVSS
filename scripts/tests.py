import sys
from functools import partial
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.transforms.fft_transform import TransformFFT
from src.transforms.mel_transform import TransformMel


def test_reconstruct_mel():
    n_fft = 1000
    hop_length = 250
    window_len = 1000
    n_mels = 80
    sample_rate = 16000
    for power in [1.0, 2.0]:
        tensor = torch.rand(1, hop_length * 10)
        transform_mel = TransformMel(
            n_fft=n_fft,
            window_len=window_len,
            hop_len=hop_length,
            power=power,
            n_mels=n_mels,
            sample_rate=sample_rate,
        )
        mel_tensor, phase = transform_mel.get_spectrogram(tensor)
        reconstructed_tensor = transform_mel.reconstruct_wav(mel_tensor, phase)
        assert reconstructed_tensor.shape == tensor.shape
        assert (reconstructed_tensor - tensor).abs().mean() < 1.0


def test_reonstruct_fft():
    n_fft = 1000
    hop_length = 250
    window_len = 1000
    power = None
    tensor = torch.rand(10, 32000)
    transform_fft = TransformFFT(
        n_fft=n_fft, window_len=window_len, hop_len=hop_length, power=power
    )
    fft_tensor, phase = transform_fft.get_spectrogram(tensor)
    reconstructed_tensor = transform_fft.reconstruct_wav(fft_tensor, phase)
    assert tensor.shape == reconstructed_tensor.shape
    assert torch.allclose(tensor, reconstructed_tensor, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    test_reconstruct_mel()
    test_reonstruct_fft()
