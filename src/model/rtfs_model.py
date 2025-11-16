from typing import Optional

import torch
from torch import nn

from src.transforms.stft_transfrom import TransformSTFT

from src.model.rtfs_parts import RTFSSeparator, RTFSDecoder, RTFSAudioEncoder, RTFSBlock, RTFSCAFBlock, VPEncoder


class RTFSModel(nn.Module):
    """
    Real-Time Full-band Speech Separation Model.
    
    The model consists of:
    - Encoder: converts audio to learned representations
    - Separator: processes representations with optional video guidance
    - Decoder: reconstructs separated audio
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
        rtfs_num_scales: int = 3,
        rtfs_sru_hidden_size: int = 128,
        rtfs_attention_heads: int = 4,
        rtfs_tfar_units: int = 2,
        # STFT parameters
        stft_n_fft: int = 512,
        stft_win_length: int = 400,
        stft_hop_length: int = 160,
        sample_rate: int = 16000,
    ):
        """
        Args:
            n_speakers: number of speakers to separate
            encoder_channels: number of channels in encoder output
            use_video: whether to use video embeddings
            tf_channels: number of channels in TF encoder/decoder
            num_rtfs_blocks: number of RTFS blocks to use
            rtfs_compressed_channels: number of compressed channels in RTFS blocks
            rtfs_num_scales: number of scales in RTFS blocks
            rtfs_sru_hidden_size: hidden size for SRU in RTFS blocks
            rtfs_attention_heads: number of attention heads in RTFS blocks
            rtfs_tfar_units: number of TF-AR reconstruction units in RTFS blocks
            stft_n_fft: number of FFT points for STFT
            stft_win_length: window length for STFT
            stft_hop_length: hop length for STFT
            sample_rate: audio sample rate
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
        )

        # Audio TF encoder/decoder around the RTFS blocks
        self.audio_encoder = RTFSAudioEncoder(
            out_channels=tf_channels,
        )

        self.ap_block = RTFSBlock(
            in_channels=tf_channels,
            compressed_channels=rtfs_compressed_channels,
            num_scales=rtfs_num_scales,
            heads=rtfs_attention_heads,
        )

        # Visual projection + CAF blocks (optional)
        if use_video:
            # Project raw video embeddings (B, T_v, D_v) -> (B, C_v, T_v)
            self.vp_block = VPEncoder(
                in_channels=tf_channels,
                compressed_channels=rtfs_compressed_channels,
                num_scales=rtfs_num_scales,
            )
            self.caf_block = RTFSCAFBlock(
                visual_channels=tf_channels,
                audio_channels=tf_channels,
                num_heads=rtfs_attention_heads,
            )

        # RTFS blocks operating on (B, C_tf, T, F)
        self.rtfs_blocks = nn.ModuleList(
            [
                RTFSBlock(
                    in_channels=tf_channels,
                    compressed_channels=rtfs_compressed_channels,
                    num_scales=rtfs_num_scales,
                    sru_hidden_size=rtfs_sru_hidden_size,
                    heads=rtfs_attention_heads,
                    tfar_units=rtfs_tfar_units,
                )
                for _ in range(num_rtfs_blocks)
            ]
        )
        
        # Separator: final 1D separation in encoder latent space.
        # We disable video fusion inside RTFSSeparator if CAF is used
        # to avoid duplicating fusion mechanisms.
        self.separator = RTFSSeparator(channels=encoder_channels)
        
        self.audio_decoder = RTFSDecoder(in_channels=tf_channels)

    def forward(
            self,
            audio_mix: torch.Tensor,
            video_embeddings: Optional[torch.Tensor] = None,
            **batch
        ):
        """
        Args:
            audio_mix: mixed audio waveform, shape (B, T)
            video_embeddings: video features corresponding to the target speaker, 
                            shape (B, T_v, D_v) or None
            **batch: other data in the batch
        Returns:
            dict with keys "audio_s{i}" of shape (B, T) for each speaker i
        """
        stft_audio_magnitude, stft_audio_phase = self.stft.get_spectrogram(audio_mix)

        # Encode audio mixture to latent space
        # stft_audio_magnitude, stft_audio_phase: (B, T, F)
        a_0 = self.audio_encoder(
            stft_audio_magnitude,
            stft_audio_phase,
        )  # (B, C_a, T_a, F_a)
        a_1 = self.ap_block(a_0)  # (B, C_a, T_a, F_a)

        # Prepare visual features if available and fuse with audio
        if self.use_video and video_embeddings is not None:
            v_1 = self.vp_block(video_embeddings)  # (B, C_v, T_v)
            tf_feats = self.caf_block(v_1, a_1)  # (B, C_a, T_a, F_a)
        else:
            tf_feats = a_1

        # Apply RTFS TF blocks
        for i, block in enumerate(self.rtfs_blocks):
            tf_feats = block(tf_feats) + a_0 # Residual connection
        # Apply separator in latent space
        z = self.separator(tf_feats)  # (B, C_a, T_a, F_a)

        # Decode to separated audio features
        stft_audio_magnitude, stft_audio_phase = self.audio_decoder(z)  # (B, C_a, T, F)

        wav = self.stft.reconstruct_wav(stft_audio_magnitude, stft_audio_phase)  # (B, T)
        wav = self._match_length(wav, audio_mix.shape[-1])

        # Apply masks and decode each speaker
        output = {
            "audio_s0": wav,
            "audio_s1": audio_mix - wav, # FIXME: Reapply all the process to second speaker
        }
        
        return output
    
    def _match_length(self, audio: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Match audio length to target length by trimming or padding.
        
        Args:
            audio: (B, T)
            target_length: target length
        Returns:
            audio: (B, target_length)
        """
        current_length = audio.shape[-1]
        
        if current_length == target_length:
            return audio
        elif current_length > target_length:
            return audio[..., :target_length]
        else:
            # Pad with zeros
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
