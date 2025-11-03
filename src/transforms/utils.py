import torch
import torchaudio


def get_spectrogram(n_fft, window_len, hop_len, n_mels, sample_rate, mode="mel_scale"):
    if (
        mode == "spec_complex"
        or mode == "spec_abs"
        or mode == "mel_complex"
        or mode == "mel_abs"
    ):

        def f(wav):
            window = torch.hann_window(window_len)
            stft = torch.stft(
                wav,
                n_fft=n_fft,
                hop_length=hop_len,
                win_length=window_len,
                window=window,
                center=False,
                onesided=True,
                return_complex=True,
            )
            if mode == "spec_complex":
                return stft

    else:
        raise NotImplementedError("Mode in get spectrogram not implemented")


def reconstruct_spec_func(n_fft, window_len, hop_len, n_mels, sample_rate):
    def f(mel_complex):
        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0.0,
            f_max=sample_rate / 2,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm=None,
        )
        mel_inv = torch.linalg.pinv(mel_fb)
        linear_complex = torch.einsum("fm,bmt->bft", mel_inv, mel_complex)
        return torch.istft(
            linear_complex,
            n_fft=n_fft,
            hop_length=256,
            win_length=1024,
            window=torch.hann_window(1024),
        )

    return f
