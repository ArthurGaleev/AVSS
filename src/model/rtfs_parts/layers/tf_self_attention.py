import torch
import torch.nn as nn
import torch.nn.functional as functional

from src.model.rtfs_parts.layers.normalization import ChannelFrequencyLayerNorm


class TFSelfAttention(nn.Module):
    """
    Time-Frequency Multi-head Self-Attention.

    Applies scaled dot-product attention independently for each frequency bin,
    allowing the model to learn temporal correlations within each frequency.
    
    Reference: https://arxiv.org/abs/2209.03952
    """
    def __init__(self, channels: int, freqencies: int, num_heads: int = 4):
        """
        Initialize TF self-attention.

        Args:
            channels: Channel dimension (D).
            freqencies: Number of frequency bins (F_comp).
            num_heads: Number of multi-head attention heads (default: 4).
        """
        super().__init__()

        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_conv = nn.Conv2d(channels, channels, kernel_size=1)

        self.act = nn.PReLU()
        self.norm = ChannelFrequencyLayerNorm(channels, freqencies)

        self.out_pathway = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.PReLU(),
            ChannelFrequencyLayerNorm(channels, freqencies),
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, D, F, T) to (B, H, F, T, D_h) for multi-head attention."""
        B, D, F, T = x.shape
        H, Dh = self.num_heads, self.head_dim
        x = x.view(B, H, Dh, F, T)
        return x.permute(0, 1, 3, 4, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, H, F, T, D_h) to (B, D, F, T)."""
        B, H, F, T, Dh = x.shape
        D = H * Dh
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        return x.view(B, D, F, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head self-attention independently for each frequency.

        Args:
            x: Features of shape (B, D, T, F).

        Returns:
            Attended features of shape (B, D, T, F).
        """
        B, D, T, F = x.shape
        x_tf = x.permute(0, 1, 3, 2)

        q = self.act(self.norm(self.q_conv(x_tf)))
        k = self.act(self.norm(self.k_conv(x_tf)))
        v = self.act(self.norm(self.v_conv(x_tf)))

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        BHF = B * self.num_heads * F
        q = q.reshape(BHF, T, self.head_dim)
        k = k.reshape(BHF, T, self.head_dim)
        v = v.reshape(BHF, T, self.head_dim)

        attn_scores = torch.bmm(q, k.transpose(1, 2))
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        attn_weights = functional.softmax(attn_scores, dim=-1)
        attn_out = torch.bmm(attn_weights, v)

        attn_out = attn_out.view(B, self.num_heads, F, T, self.head_dim)
        attn_out = self._merge_heads(attn_out)

        out = self.out_pathway(attn_out)
        out = out.permute(0, 1, 3, 2)
      
        return out
