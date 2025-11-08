from functools import partial

import torch
import torchaudio


class TransformMel(torch.nn.Module):
    def __init__(
        self,
        n_fft,
        window_len,
        hop_len,
        n_mels,
        sample_rate,
        power=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if power is not None:
            assert power != 0, "Power isnt equal to zero"
        self.get_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            hop_length=hop_len,
            win_length=window_len,
            n_fft=n_fft,
            n_mels=n_mels,
            power=power,
        )
        self.kwargs = {
            "n_fft": n_fft,
            "win_length": window_len,
            "hop_length": hop_len,
        }
        self.reconstruct = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=sample_rate
        )
        self.power = power

    def get_spectrogram(self, wav):
        assert (
            wav.shape[-1] % self.kwargs["hop_length"] == 0
        ), "Wav should have length that is divisable by hop length for Mel. Otherwise the shape wont be preserved"
        window = torch.hann_window(self.kwargs["n_fft"])
        phase = torch.stft(
            wav, window=window.to(wav.device), **self.kwargs, return_complex=True
        ).imag
        return self.get_spec(wav), phase

    def reconstruct_wav(self, spec, phase=None):
        spec = self.reconstruct(spec)
        if self.power is not None:
            spec = spec ** (1 / self.power)
        if phase is not None:
            spec = spec * torch.exp(1j * phase)
        else:
            spec = spec.to(torch.complex64)
        return torch.istft(
            spec,
            window=torch.hann_window(self.kwargs["n_fft"]).to(spec.device),
            **self.kwargs,
        )
