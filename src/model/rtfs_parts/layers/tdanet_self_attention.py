import torch
import torch.nn as nn

from src.model.rtfs_parts.layers.normalization import GlobalLayerNorm


class FFN(nn.Module):
    """
    Feed-Forward Network (FFN) module.
    """
    def __init__(self, channels: int, expansion_factor: int = 2, dropout: float = 0.1):
        super().__init__()

        self.channels = channels
        self.hidden_dim = channels * expansion_factor
        self.dropout = dropout

        self.ffn = nn.Sequential(
            nn.Conv1d(channels, self.hidden_dim, kernel_size=1, bias=False),
            GlobalLayerNorm(self.hidden_dim),
            nn.PReLU(),
            nn.Dropout(p=self.dropout),
            nn.Conv1d(self.hidden_dim, channels, kernel_size=5, padding=2, groups=channels),  # depthwise conv
            GlobalLayerNorm(channels),
            nn.PReLU(),
            nn.Dropout(p=self.dropout),
            nn.Conv1d(self.hidden_dim, channels, kernel_size=1, bias=False),
            GlobalLayerNorm(channels),
            nn.PReLU(),
            nn.Dropout(p=self.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, T, F)
        Returns:
            (B, D, T, F)
        """
        return self.ffn(x)

class TDANetSelfAttention(nn.Module):
    """
    TDANet Self-attention.

    See https://arxiv.org/abs/2209.15200 for reference.
    """
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.mhsa = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.ffn = FFN(channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply full-band self-attention.

        Args:
            x: (B, D, T)

        Returns:
            (B, D, T)
        """
        B, D, T = x.shape

        # Multi-head self-attention
        attn_out, _ = self.mhsa(x.transpose(1, 2), x.transpose(1, 2), x.transpose(1, 2))  # (B, T, D)
        attn_out = attn_out.transpose(1, 2)  # (B, D, T)
        attn_out = attn_out + x  # Residual

        # Feed-forward network
        ffn_out = self.ffn(attn_out)  # (B, D, T)
        ffn_out = ffn_out + attn_out  # Residual

        return ffn_out
