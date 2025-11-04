import sys
from functools import partial
from pathlib import Path

import torch
import torchaudio

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.transforms.spec_utils import GetSpectrogram, Reconstruct


def test_reconstruct():
    sample_rate = 16000
    n_fft = 1000
    hop_length = 250
    window_len = 1000
    n_mels = 80
    get_spectrogram_ = partial(
        GetSpectrogram,
        n_fft=n_fft,
        window_len=window_len,
        hop_len=hop_length,
        n_mels=n_mels,
        sample_rate=sample_rate,
    )
    reconstruct_ = partial(
        Reconstruct,
        n_fft=n_fft,
        window_len=window_len,
        hop_len=hop_length,
        n_mels=n_mels,
        sample_rate=sample_rate,
    )
    power = None
    mode = None
    tensor = torch.rand(10, 32000)
    reconstructed_wav = reconstruct_(mode=mode, power=power)(
        get_spectrogram_(mode=mode, power=power)(tensor)[0]
    )
    assert tensor.shape == reconstructed_wav.shape
    assert torch.allclose(tensor, reconstructed_wav, rtol=1e-4, atol=1e-5)
    mode = "mel"
    window = torch.hann_window(n_fft)
    for power in [1.0, 2.0]:
        tensor = torch.rand(1, hop_length * 10)
        phase = torch.stft(
            tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=window_len,
            window=window,
            return_complex=True,
        ).imag
        mel_tensor, phase = get_spectrogram_(mode=mode, power=power)(tensor)
        reconstructed_tensor = reconstruct_(mode=mode, power=power)(mel_tensor, phase)
        assert reconstructed_tensor.shape == tensor.shape
        assert (reconstructed_tensor - tensor).abs().mean() < 1.0


if __name__ == "__main__":
    test_reconstruct()
