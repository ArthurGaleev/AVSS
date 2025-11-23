from typing import List, Optional

import torch
from torch import nn
from torchvision.transforms.v2 import Transform

from src.model.rtfs_parts import (
    RTFSAudioEncoder,
    RTFSBlock,
    RTFSCAFBlock,
    RTFSDecoder,
    RTFSSeparator,
    VPEncoder,
)
from src.transforms.stft_transfrom import TransformSTFT


class RTFSModel(nn.Module):
    """
    Real-Time Full-band Speech Separation Model.

    Implements RTFS-Net architecture for speaker separation. The model:
    1. Encodes mixed audio spectrogram to latent representation via STFT and audio encoder
    2. Optionally processes video embeddings and fuses with audio via CAF blocks
    3. Applies stacked RTFS blocks for audio-visual feature refinement
    4. Separates speakers using learnable masks in latent space
    5. Decodes separated features back to time-domain waveforms

    Reference: https://arxiv.org/abs/2309.17189
    """

    def __init__(
        self,
        n_speakers: int = 2,
        encoder_channels: int = 256,
        use_video: bool = True,
        # RTFS-specific hyperparameters
        tf_channels: int = 256,
        num_rtfs_blocks: int = 2,
        rtfs_compressed_channels: int = 128,
        rtfs_sru_hidden_size: int = 128,
        rtfs_attention_heads: int = 4,
        rtfs_tfar_units: int = 2,
        reuse_rtfs_blocks: bool = True,
        # STFT parameters
        stft_n_fft: int = 512,
        stft_win_length: int = 400,
        stft_hop_length: int = 160,
        sample_rate: int = 16000,
        transforms: List[Transform] | None = None,
    ):
        """
        Initialize RTFS-Net model.

        Args:
            n_speakers: Number of speakers to separate (default: 2).
            encoder_channels: Dimension of video/auxiliary embeddings (default: 256).
            use_video: Whether to use video embeddings for audio-visual fusion (default: True).
            tf_channels: Number of channels in time-frequency domain (C_a) (default: 256).
            num_rtfs_blocks: Number of stacked RTFS blocks (default: 2).
            rtfs_compressed_channels: Channel dimension after compression (D) (default: 128).
            rtfs_sru_hidden_size: Hidden state size for SRU in RTFS blocks (default: 128).
            rtfs_attention_heads: Number of multi-head attention heads (default: 4).
            rtfs_tfar_units: Number of TF-AR reconstruction scales (default: 2).
            reuse_rtfs_blocks: Whether to reuse weights across RTFS blocks (default: True).
            stft_n_fft: FFT size for STFT transform (default: 512).
            stft_win_length: Window length for STFT (default: 400).
            stft_hop_length: Hop length for STFT (default: 160).
            sample_rate: Audio sample rate in Hz (default: 16000).
            transforms: Optional list of spectral augmentations (default: None).
        """
        super().__init__()

        self.n_speakers = n_speakers
        self.use_video = use_video

        # Save some RTFS-specific sizes for shape handling
        self.encoder_channels = encoder_channels
        self.tf_channels = tf_channels
        self.num_rtfs_blocks = num_rtfs_blocks

        self.stft = TransformSTFT(
            n_fft=stft_n_fft,
            window_len=stft_win_length,
            hop_len=stft_hop_length,
            sample_rate=sample_rate,
            transforms=transforms,
        )
        self.audio_encoder = RTFSAudioEncoder(
            out_channels=tf_channels,
        )

        self.ap_block = RTFSBlock(
            in_channels=tf_channels,
            compressed_channels=rtfs_compressed_channels,
            downsample_units=rtfs_tfar_units,
            unfold_kernel_size=8,
            sru_hidden_size=rtfs_sru_hidden_size,
            sru_num_layers=4,
            attention_heads=rtfs_attention_heads,
            freqencies=stft_n_fft // 2 + 1,
        )

        if use_video:
            self.vp_block = VPEncoder(
                in_channels=encoder_channels,
                compressed_channels=rtfs_compressed_channels,
                downsample_units=rtfs_tfar_units,
                heads=rtfs_attention_heads,
            )
            self.caf_block = RTFSCAFBlock(
                visual_channels=encoder_channels,
                audio_channels=tf_channels,
                num_heads=rtfs_attention_heads,
            )
        self.rtfs_blocks = nn.ModuleList(
            [
                (
                    RTFSBlock(
                        in_channels=tf_channels,
                        compressed_channels=rtfs_compressed_channels,
                        downsample_units=rtfs_tfar_units,
                        unfold_kernel_size=8,
                        sru_hidden_size=rtfs_sru_hidden_size,
                        sru_num_layers=4,
                        attention_heads=rtfs_attention_heads,
                        freqencies=stft_n_fft // 2 + 1,
                    )
                    if not reuse_rtfs_blocks
                    else self.ap_block
                )
                for _ in range(num_rtfs_blocks)
            ]
        )

        self.separator = RTFSSeparator(channels=tf_channels)
        self.audio_decoder = RTFSDecoder(in_channels=tf_channels)

    def forward(
        self,
        audio_mix: torch.Tensor,
        video_embeddings: Optional[torch.Tensor] = None,
        **batch,
    ):
        """
        Separate mixed audio into multiple speakers with optional video guidance.

        Args:
            audio_mix: Mixed audio waveform of shape (B, T_audio).
            video_embeddings: Video embeddings for each speaker of shape (B, S, C_v, T_video),
                or None if audio-only separation (default: None).
            **batch: Additional batch data (unused).

        Returns:
            Dictionary containing:
                - audio_s{i}: Separated waveform for speaker i, shape (B, T_audio)
                - spectrogram_s{i}: Input spectrogram magnitude, shape (B, T_spec, F)
                - spectrogram_pred_s{i}: Predicted magnitude for speaker i, shape (B, T_spec, F)
        """
        output = {}

        # Prepare visual features if available and fuse with audio
        if self.use_video:
            B, S, D_v, T_v = video_embeddings.shape
            video_embeddings = video_embeddings.view(B * S, D_v, T_v)

            v_1 = self.vp_block(video_embeddings)
            
            _, D_v, T_v = v_1.shape
            v_1 = v_1.reshape(B, S, D_v, T_v)
        else:
            v_1 = None

        for speaker_idx in range(self.n_speakers):
            stft_audio_magnitude, stft_audio_phase = self.stft.get_spectrogram(audio_mix)
            output[f"spectrogram_s{speaker_idx}"] = stft_audio_magnitude

            a_0 = self.audio_encoder(stft_audio_magnitude, stft_audio_phase)
            a_1 = self.ap_block(a_0)
            
            if v_1 is not None:
                tf_feats = self.caf_block(v_1[:, speaker_idx, :, :], a_1)
            else:
                tf_feats = a_1

            for block in self.rtfs_blocks:
                tf_feats = block(tf_feats) + a_0

            z = self.separator(tf_feats, a_0)
            
            stft_audio_magnitude, stft_audio_phase = self.audio_decoder(z)
            
            wav = self.stft.reconstruct_wav(stft_audio_magnitude, stft_audio_phase)
            wav = self._match_length(wav, audio_mix.shape[-1])

            output[f"audio_s{speaker_idx}"] = wav
            output[f"spectrogram_pred_s{speaker_idx}"] = stft_audio_magnitude

            # Remove speaker from mixture for next iteration
            audio_mix = audio_mix - wav

        return output

    def _match_length(self, audio: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Match audio length to target by trimming or zero-padding.

        Args:
            audio: Audio waveform of shape (B, T).
            target_length: Target length in samples.

        Returns:
            Audio waveform of shape (B, target_length).
        """
        current_length = audio.shape[-1]

        if current_length == target_length:
            return audio
        elif current_length > target_length:
            return audio[..., :target_length]
        else:
            padding = target_length - current_length
            return torch.nn.functional.pad(audio, (0, padding))

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
