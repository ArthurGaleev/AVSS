import torch
from torch import nn
import torch.nn.functional as functional


class RTFSCAFBlock(nn.Module):
    """
    Cross-dimensional Attention Fusion Block.
    Fuses visual and auditory embeddings using attention and gated fusion mechanisms.
    """

    def __init__(
        self,
        visual_channels: int,
        audio_channels: int,
        num_heads: int = 4,
        groups_attention: int = 1,
        groups_gated: int = 1,
    ):
        """
        Args:
            visual_channels: Number of channels in visual embedding (C_v)
            audio_channels: Number of channels in audio embedding (C_a)
            num_heads: Number of attention heads (h)
            groups_attention: Number of groups for attention convolution (C_1)
            groups_gated: Number of groups for gated convolution (C_2)
        """
        super().__init__()
        self.num_heads = num_heads
        self.visual_channels = visual_channels
        self.audio_channels = audio_channels

        # Attention Fusion components (F1)
        self.visual_attention_pathway = nn.Sequential(
            nn.Conv1d(
                visual_channels,
                audio_channels * num_heads,
                kernel_size=1,
                groups=groups_attention,
            ), # group convolution (B, C_v * h, T_v)
            nn.GroupNorm(num_groups=1, num_channels=audio_channels * num_heads), # gLN (B, C_a * h, T_v)
            nn.AvgPool2d(kernel_size=(2,1), stride=(2,1)), # mean pooling across heads (B, C_a, T_v)
            nn.Softmax(dim=-1), # softmax to get attention weights (B, C_a, T_v)
        )
        self.audio_attention_pathway = nn.Sequential(
            nn.Conv2d(
                audio_channels,
                audio_channels,
                kernel_size=1,
                groups=groups_attention,
            ), # depthwise convolution (B, C_a, T_a, F)
            nn.GroupNorm(num_groups=1, num_channels=audio_channels), # gLN (B, C_a, T_a, F)
        )

        # Gated Fusion components (F2)
        self.visual_gated_pathway = nn.Sequential(
            nn.Conv1d(
                visual_channels,
                audio_channels,
                kernel_size=1,
                groups=groups_gated,
            ), # group convolution (B, C_a, T_v)
            nn.GroupNorm(num_groups=1, num_channels=audio_channels), # gLN (B, C_a, T_v)
        )
        self.audio_gated_pathway = nn.Sequential(
            nn.Conv2d(
                audio_channels,
                audio_channels,
                kernel_size=1,
                groups=groups_gated,
            ), # depthwise convolution (B, C_a, T_a, F)
            nn.GroupNorm(num_groups=1, num_channels=audio_channels), # gLN (B, C_a, T_a, F)
            nn.ReLU(), # ReLU activation
        )

    def forward(
        self,
        visual_embedding: torch.Tensor,
        auditory_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the RTFS CAF Block.
        Args:
            visual_embedding (torch.Tensor): Visual embedding tensor of shape (B, C_a, T_v).
            auditory_embedding (torch.Tensor): Auditory embedding tensor of shape (B, C_a, T_a, F).
        Returns:
            torch.Tensor: Fused auditory embedding of shape (B, C_a, T_a, F).
        """
        _, _, T_a, _ = auditory_embedding.shape
        _, _, T_v = visual_embedding.shape

        # ===== Attention Fusion (F1) =====
        a_val = self.audio_attention_pathway(auditory_embedding)  # (B, C_a, T_a, F)
        v_attn = self.visual_attention_pathway(visual_embedding)  # (B, C_a, T_v)

        # align T_v with T_a using interpolation)
        if T_v != T_a:
            v_attn_aligned = functional.interpolate(v_attn, size=T_a, mode='nearest')  # (B, C_a, T_a)
        else:
            v_attn_aligned = v_attn

        f_1 = a_val * v_attn_aligned.unsqueeze(-1)  # (B, C_a, T_a, F)

        # ===== Gated Fusion (F2) =====
        a_gate = self.audio_gated_pathway(auditory_embedding)  # (B, C_a, T_a, F)
        v_key = self.visual_gated_pathway(visual_embedding)  # (B, C_a, T_v)
        
        # align T_v with T_a using interpolation)
        if T_v != T_a:
            v_key_aligned = functional.interpolate(v_key, size=T_a, mode='nearest')  # (B, C_a, T_a)
        else:
            v_key_aligned = v_key

        f_2 = a_gate * v_key_aligned.unsqueeze(-1)  # (B, C_a, T_a, F)
        
        # Sum the two fused features
        a_2 = f_1 + f_2  # (B, C_a, T_a, F)

        return a_2
