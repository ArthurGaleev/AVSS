import torch
import torch.nn as nn

from src.model.rtfs_parts.layers.normalization import GlobalLayerNorm


class FFN(nn.Module):
    """
    Feed-Forward Network for transformer-based architectures.

    Expands features via point-wise convolutions with gating and normalization.
    """
    def __init__(self, channels: int, expansion_factor: int = 2, dropout: float = 0.1):
        """
        Initialize feed-forward network.

        Args:
            channels: Input/output channel dimension.
            expansion_factor: Expansion factor for hidden dimension (default: 2).
            dropout: Dropout probability (default: 0.1).
        """
        super().__init__()

        self.channels = channels
        self.hidden_dim = channels * expansion_factor
        self.dropout = dropout

        self.ffn = nn.Sequential(
            nn.Conv1d(channels, self.hidden_dim, kernel_size=1, bias=False),
            GlobalLayerNorm(self.hidden_dim),
            nn.PReLU(),
            nn.Dropout(p=self.dropout),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, padding=2, groups=self.hidden_dim),
            GlobalLayerNorm(self.hidden_dim),
            nn.PReLU(),
            nn.Dropout(p=self.dropout),
            nn.Conv1d(self.hidden_dim, channels, kernel_size=1, bias=False),
            GlobalLayerNorm(channels),
            nn.PReLU(),
            nn.Dropout(p=self.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward transformation.

        Args:
            x: Features of shape (B, D, T).

        Returns:
            Transformed features of shape (B, D, T).
        """
        return self.ffn(x)

class TDANetSelfAttention(nn.Module):
    """
    TDANet: Transformer with Dual-path Architecture Self-Attention.

    Combines multi-head self-attention and feed-forward networks with residual
    connections for temporal sequence modeling.

    Reference: https://arxiv.org/abs/2209.15200
    """
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize TDANet self-attention.

        Args:
            channels: Channel dimension (D).
            num_heads: Number of multi-head attention heads (default: 4).
            dropout: Dropout probability (default: 0.1).
        """
        super().__init__()
        
        self.mhsa = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.ffn = FFN(channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer block with attention and feed-forward.

        Args:
            x: Features of shape (B, D, T).

        Returns:
            Transformed features of shape (B, D, T).
        """
        B, D, T = x.shape
        
        attn_out, _ = self.mhsa(x.transpose(1, 2), x.transpose(1, 2), x.transpose(1, 2))
        attn_out = attn_out.transpose(1, 2)
        attn_out = attn_out + x

        ffn_out = self.ffn(attn_out)
        ffn_out = ffn_out + attn_out

        return ffn_out
