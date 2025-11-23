from typing import Dict, Optional

import torch
import torch.nn as nn


class SeparationEnhancer(nn.Module):
    """
    Separation Enhancer module for refining separated audios by reapplying model once again.
    """
    def __init__(self):
        """
        Initialize Separation Enhancer.
        """
        super().__init__()

    def forward(
            self,
            model: nn.Module,
            audio_s0: torch.Tensor,
            audio_s1: torch.Tensor,
            video_embeddings: Optional[torch.Tensor] = None,
            **batch
        ) -> Dict[str, torch.Tensor]:
        """
        Enhance separated audios by reapplying the model.
        Args:
            model: The RTFS model to reapply.
            audio_s0: First separated audio of shape (B, T_audio).
            audio_s1: Second separated audio of shape (B, T_audio).
            video_embeddings: Video embeddings for each speaker of shape (B, S, C_v, T_video),
                or None if audio-only separation (default: None).
            **batch: Additional batch data (unused).
        Returns:
            Dictionary with enhanced separated audios.
        """

        first_video_embeddings = video_embeddings
        second_video_embeddings = None if video_embeddings is None else video_embeddings[:, ::-1, :, :]

        enhanced_first = model(
            audio_mix=audio_s0,
            video_embeddings=first_video_embeddings,
            **batch
        )

        enhanced_second = model(
            audio_mix=audio_s1,
            video_embeddings=second_video_embeddings,
            **batch
        )

        return {
            "audio_s0": enhanced_first["audio_s0"] + enhanced_second["audio_s1"],
            "audio_s1": enhanced_first["audio_s1"] + enhanced_second["audio_s0"],
        }
