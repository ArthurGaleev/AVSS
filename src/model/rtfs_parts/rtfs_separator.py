import torch
from torch import nn


class RTFSSeparator(nn.Module):
    """
    Separator network for RTFS model.
    Processes latent representations with optional video guidance to produce separation masks.
    """

    def __init__(
        self,
        channels: int,
    ):
        """
        Args:
            channels: number of channels
        """
        super().__init__()
        
        self.mask_pathway = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.ReLU(),
        )

    def forward(
        self,
        a_R: torch.Tensor,
        a_0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            a_R: (B, C_a, T, F) intermediate representation
            a_0: (B, C_a, T, F) original encoded features
        Returns:
            output: (B, C_a, T, F)
        """
        m = self.mask_pathway(a_R) # (B, C_a, T, F)

        m_r, m_i = torch.chunk(m, 2, dim=1)  # (B, C_a/2, T, F) each
        E_r, E_i = torch.chunk(a_0, 2, dim=1)  # (B, C_a/2, T, F) each

        z_r = m_r * E_r - m_i * E_i  # (B, C_a/2, T, F)
        z_i = m_r * E_i + m_i * E_r  # (B, C_a/2, T, F)

        output = torch.cat((z_r, z_i), dim=1)  # (B, C_a, T, F)
        return output
