from functools import partial

import torch
import torchaudio


class GetSpectrogram(torch.nn.Module):
    def __init__(
        self,
        n_fft,
        window_len,
        hop_len,
        n_mels,
        sample_rate,
        mode=None,
        power=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if mode == "mel":
            self.get_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                hop_length=hop_len,
                win_length=window_len,
                n_fft=n_fft,
                n_mels=n_mels,
                power=power,
            )
        else:
            self.get_spec = torchaudio.transforms.Spectrogram(
                hop_length=hop_len, win_length=window_len, n_fft=n_fft, power=power
            )
        self.stft_kwargs = {
            "n_fft": n_fft,
            "win_length": window_len,
            "hop_length": hop_len,
        }

    def forward(self, wav):
        assert (
            wav.shape[-1] % self.stft_kwargs["hop_length"] == 0
        ), "Wav should have length that is divisable by hop length. Otherwise the shape wont be preserved"
        window = torch.hann_window(self.stft_kwargs["n_fft"])
        phase = torch.stft(
            wav, window=window.to(wav.device), **self.stft_kwargs, return_complex=True
        ).imag
        return self.get_spec(wav), phase


class Reconstruct(torch.nn.Module):
    def __init__(
        self,
        n_fft,
        window_len,
        hop_len,
        n_mels,
        sample_rate,
        mode=None,
        power=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if power is not None:
            assert power != 0, "Power isnt equal to zero"
        if mode == "mel":
            self.reconstruct = torchaudio.transforms.InverseMelScale(
                n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=sample_rate
            )
        elif power is None:
            self.reconstruct = torchaudio.transforms.InverseSpectrogram(
                n_fft=n_fft, hop_length=hop_len, win_length=window_len
            )
        else:
            self.reconstruct = torch.nn.Identity()
        self.power = power
        self.mode = "mel"
        self.istft_kwargs = {
            "n_fft": n_fft,
            "win_length": window_len,
            "hop_length": hop_len,
        }

    def forward(self, spec, phase=None):
        spec = self.reconstruct(spec)
        if self.power is not None:
            spec = spec ** (1 / self.power)
        if phase is None:
            return spec
        spec = spec * torch.exp(1j * phase)
        window = torch.hann_window(self.istft_kwargs["n_fft"])
        return torch.istft(
            spec,
            window=window.to(spec.device),
            **self.istft_kwargs,
        )
