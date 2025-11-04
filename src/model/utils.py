import torch
import torchaudio


def reconstruct_spec_func(n_fft, window_len, hop_len, n_mels, sample_rate):
    def f(spec):
        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0.0,
            f_max=sample_rate / 2,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm=None,
        )
        mel_inv = torch.linalg.pinv(mel_fb)
        mel_complex = torch.randn(1, n_mels, 200, dtype=torch.complex64)
        linear_complex = torch.einsum("fm,bmt->bft", mel_inv, mel_complex)
        return torch.istft(
            linear_complex,
            n_fft=n_fft,
            hop_length=256,
            win_length=1024,
            window=torch.hann_window(1024),
        )

    return f
