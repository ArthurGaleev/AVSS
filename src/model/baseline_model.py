from torch import nn
from torch.nn import Sequential


class BaselineModel(nn.Module):
    """
    Simple MLP
    """

    def __init__(self, n_feats, reconstruct_spec_func=None, fc_hidden=512):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        self.net = Sequential(
            nn.Linear(in_features=n_feats, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=2 * n_feats),
        )

    def forward(self, spectrogram_mix, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """

        output = self.net(spectrogram_mix.real.transpose(-1, -2)).transpose(-1, -2)
        spec1, spec2 = output.chunk(2, dim=-2)
        return {
            "spectrogram_pred_first": spec1,
            "spectrogram_pred_second": spec2,
        }
