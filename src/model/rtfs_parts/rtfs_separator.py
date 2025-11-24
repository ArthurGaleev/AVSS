import torch
from torch import nn


class RTFSSeparator(nn.Module):
    """
    RTFS Separator: produces speaker separation masks in latent space.

    Processes intermediate features to generate complex-valued masks that
    separate speakers by learning phase and magnitude transformations.
    """

    def __init__(
        self,
        channels: int,
    ):
        """
        Initialize separator.

        Args:
            channels: Channel dimension (C_a).
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
        Separate speakers via complex-valued mask multiplication.

        Args:
            a_R: Intermediate representation of shape (B, C_a, T, F).
            a_0: Original encoded features of shape (B, C_a, T, F).

        Returns:
            Separated features of shape (B, C_a, T, F).
        """
        m = self.mask_pathway(a_R)

        m_r, m_i = torch.chunk(m, 2, dim=1)
        E_r, E_i = torch.chunk(a_0, 2, dim=1)

        z_r = m_r * E_r - m_i * E_i
        z_i = m_r * E_i + m_i * E_r

        output = torch.cat((z_r, z_i), dim=1)
        
        return output
