from functools import partial

import torch
import torchaudio


class TransformFFT(torch.nn.Module):
    def __init__(
        self,
        n_fft,
        window_len,
        hop_len,
        power=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if power is not None:
            assert power != 0, "Power isnt equal to zero"
        self.get_spec = torchaudio.transforms.Spectrogram(
            hop_length=hop_len, win_length=window_len, n_fft=n_fft, power=None
        )
        self.reconstruct = torchaudio.transforms.InverseSpectrogram(
            n_fft=n_fft, hop_length=hop_len, win_length=window_len
        )
        self.power = power

    def get_spectrogram(self, wav):
        spec = self.get_spec(wav)
        if self.power is not None:
            return spec.real ** (self.power), spec.imag
        return spec.real, spec.imag

    def reconstruct_wav(self, spec, phase=None):
        if self.power is not None:
            spec = spec ** (1 / self.power)
        if phase is not None:
            spec = spec * torch.exp(1j * phase)
        return self.reconstruct(spec)
