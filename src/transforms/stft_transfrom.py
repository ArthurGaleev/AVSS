import torch


class TransformSTFT(torch.nn.Module):
    def __init__(
        self,
        n_fft,
        window_len,
        hop_len,
        sample_rate,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.kwargs = {
            "n_fft": n_fft,
            "win_length": window_len,
            "hop_length": hop_len,
        }
        self.sample_rate = sample_rate

    def get_spectrogram(self, wav):
        assert (
            wav.shape[-1] % self.kwargs["hop_length"] == 0
        ), "Wav should have length that is divisable by hop length for STFT. Otherwise the shape wont be preserved"
        window = torch.hann_window(self.kwargs["win_length"])
        stft_result = torch.stft(
            wav, window=window.to(wav.device), **self.kwargs, return_complex=True
        )
        return stft_result.abs(), stft_result.angle()

    def reconstruct_wav(self, spec, phase=None):
        if phase is not None:
            spec = spec * torch.exp(1j * phase)
        else:
            spec = spec.to(torch.complex64)
        return torch.istft(
            spec,
            window=torch.hann_window(self.kwargs["win_length"]).to(spec.device),
            **self.kwargs,
        )
