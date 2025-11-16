import torch
import torch.nn as nn
import torch.nn.functional as functional

from src.model.rtfs_parts.layers.normalization import ChannelFrequencyLayerNorm


class TFSelfAttention(nn.Module):
    """
    Self-attention over the TF domain.

    See https://arxiv.org/abs/2209.03952 for reference.
    """
    def __init__(self, channels: int, freqencies: int, num_heads: int = 4):
        super().__init__()

        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # Head-specific 1x1 convs to generate Q, K, V features
        self.q_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_conv = nn.Conv2d(channels, channels, kernel_size=1)

        self.act = nn.PReLU()
        self.norm = ChannelFrequencyLayerNorm(channels, freqencies)

        # Final projection back to channels with nonlinear + norm
        self.out_pathway = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.PReLU(),
            ChannelFrequencyLayerNorm(channels, freqencies),
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, D, F, T) -> (B, H, F, T, D_h)."""
        B, D, F, T = x.shape
        H, Dh = self.num_heads, self.head_dim
        x = x.view(B, H, Dh, F, T)
        # (B, H, Dh, F, T) -> (B, H, F, T, Dh)
        return x.permute(0, 1, 3, 4, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, F, T, D_h) -> (B, D, F, T)."""
        B, H, F, T, Dh = x.shape
        D = H * Dh
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # (B, H, Dh, F, T)
        return x.view(B, D, F, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply full-band self-attention.

        Args:
            x: (B, D, T, F)

        Returns:
            (B, D, T, F)
        """
        B, D, T, F = x.shape

        # Work in (B, D, F, T) where attention is over T for each F
        x_tf = x.permute(0, 1, 3, 2)  # (B, D, F, T)

        # Generate Q, K, V via 1x1 convs with nonlinearity + norm
        q = self.act(self.norm(self.q_conv(x_tf)))  # (B, D, F, T)
        k = self.act(self.norm(self.k_conv(x_tf)))
        v = self.act(self.norm(self.v_conv(x_tf)))

        # Split into heads
        q = self._split_heads(q)  # (B, H, F, T, Dh)
        k = self._split_heads(k)  # (B, H, F, T, Dh)
        v = self._split_heads(v)  # (B, H, F, T, Dh)

        # Reshape to merge batch and freq for efficient attention
        BHF = B * self.num_heads * F
        q = q.reshape(BHF, T, self.head_dim)  # (B*H*F, T, Dh)
        k = k.reshape(BHF, T, self.head_dim)
        v = v.reshape(BHF, T, self.head_dim)

        # Scaled dot-product attention over time dimension
        attn_scores = torch.bmm(q, k.transpose(1, 2))  # (BHF, T, T)
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        attn_weights = functional.softmax(attn_scores, dim=-1)
        attn_out = torch.bmm(attn_weights, v)  # (BHF, T, Dh)

        # Restore head and freq dimensions
        attn_out = attn_out.view(B, self.num_heads, F, T, self.head_dim)  # (B,H,F,T,Dh)
        attn_out = self._merge_heads(attn_out)  # (B,D,F,T)

        # Back to (B, D, T, F)
        attn_out = attn_out.permute(0, 1, 3, 2)

        # Final projection
        out = self.out_pathway(attn_out)  # (B, D, T, F)
        return out
