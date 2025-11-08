import math

import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_speakers):
        super().__init__()

        self.dconv1d_block = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1),
            nn.PReLU(),
            nn.GroupNorm(
                num_groups=1, num_channels=hidden_dim
            ),  # TODO: add cumulative layer norm, because the task is causal
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim,
            ),
            nn.PReLU(),
            nn.GroupNorm(
                num_groups=1, num_channels=hidden_dim
            ),  # TODO: add cumulative layer norm, because the task is causal
        )

        self.res_conv = nn.Conv1d(
            in_channels=hidden_dim, out_channels=hidden_dim * n_speakers, kernel_size=1
        )
        self.out_conv = nn.Conv1d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1
        )

        # self.linear = nn.Linear(input_dim, hidden_dim)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, audio_mix):
        x = self.dconv1d_block(
            audio_mix.unsqueeze(1)
        )  # expected num_channels=1 in conv1d

        residual = self.res_conv(x)
        x = self.out_conv(x) + audio_mix.unsqueeze(1)

        # x = self.linear(audio_mix)
        # x = self.sigmoid()

        return residual, x


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.dconv1d_block = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=input_dim, out_channels=input_dim, kernel_size=1
            ),
            nn.PReLU(),
            nn.GroupNorm(
                num_groups=1, num_channels=input_dim
            ),  # TODO: add cumulative layer norm, because the task is causal
            nn.ConvTranspose1d(
                in_channels=input_dim,
                out_channels=input_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim,
            ),
            nn.PReLU(),
            nn.GroupNorm(
                num_groups=1, num_channels=input_dim
            ),  # TODO: add cumulative layer norm, because the task is causal
        )

        self.out_conv = nn.ConvTranspose1d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=1
        )

    def forward(self, x):
        x = self.dconv1d_block(x)

        x = self.out_conv(x)

        return x.squeeze(1)


class Segmentation(nn.Module):
    def __init__(self, n_frames, overlap=True):
        super().__init__()
        self.n_frames = n_frames
        self.overlap = overlap

    def forward(self, x):
        # as in original DPRNN paper
        # chunk_size = √(2*n_frames)
        chunk_size = int(math.sqrt(2 * self.n_frames))

        if self.overlap:
            hop_size = chunk_size // 2  # 50% overlap
        else:
            hop_size = chunk_size  # no overlap

        # the first and last samples are padded to chunk_size
        x = F.pad(x, (chunk_size // 2, chunk_size - (self.n_frames % chunk_size) - 1))

        x = x.unfold(2, chunk_size, hop_size)

        # need to permute stacked chunks of shape
        # (batch_size x n_features x n_chunks x chunk_size) -> (batch_size x n_features x chunk_size x n_chunks)
        return x.permute(0, 1, 3, 2)


class OverlapAdd(nn.Module):
    def __init__(self, n_frames, overlap=True):
        super().__init__()
        self.n_frames = n_frames
        self.overlap = overlap

    def forward(self, x):
        batch, features, chunk_size, n_chunks = x.shape

        if self.overlap:
            hop_size = chunk_size // 2
        else:
            hop_size = chunk_size

        output_length = (n_chunks - 1) * hop_size + chunk_size

        # Inverse to unfold
        x = F.fold(
            x.contiguous().view(batch, features * chunk_size, n_chunks),
            output_size=(1, output_length),
            kernel_size=(1, chunk_size),
            stride=(1, hop_size),
        )

        # reshape back and cut extra frames
        return x.contiguous().view(batch, features, output_length)[
            :, :, : self.n_frames
        ]


class DPRNNBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.intra_rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.intra_fc = nn.Linear(in_features=2 * hidden_dim, out_features=input_dim)
        self.intra_norm = nn.LayerNorm(input_dim)

        self.inter_rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.inter_fc = nn.Linear(in_features=2 * hidden_dim, out_features=input_dim)
        self.inter_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        batch_size, n_features, chunk_size, n_chunks = x.shape

        # because intra_rnn is applied to individual chunks
        x = (
            x.permute(0, 3, 2, 1)
            .contiguous()
            .view(batch_size * n_chunks, chunk_size, n_features)
        )
        x = self.intra_norm(self.intra_fc(self.intra_rnn(x)[0])) + x

        # because inter_rnn is applied across the chunks
        x = x.contiguous().view(batch_size * chunk_size, n_chunks, n_features)
        x = self.inter_norm(self.inter_fc(self.inter_rnn(x)[0])) + x

        # reshape back
        return x.contiguous().view(batch_size, n_features, chunk_size, n_chunks)


class DPRNN(nn.Module):
    def __init__(self, n_frames, input_dim, hidden_dim, n_dprnn_blocks, overlap=True):
        super().__init__()

        self.segmentation = Segmentation(n_frames=n_frames, overlap=overlap)

        self.dprnn_blocks = nn.ModuleList(
            [
                DPRNNBlock(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                )
                for _ in range(n_dprnn_blocks)
            ]
        )

        self.overlap_add = OverlapAdd(n_frames=n_frames, overlap=overlap)

    def forward(self, x):
        x = self.segmentation(x)

        for block in self.dprnn_blocks:
            x = block(x)

        x = self.overlap_add(x)

        return x


class DPRNN_TasNet(nn.Module):
    def __init__(
        self,
        n_frames,
        input_dim=1,
        enc_hidden_dim=64,
        rnn_hidden_dim=128,
        n_dprnn_blocks=6,
        n_speakers=2,
        overlap=True,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.enc_hidden_dim = enc_hidden_dim
        self.n_speakers = n_speakers

        self.encoder = Encoder(
            input_dim=input_dim, hidden_dim=enc_hidden_dim, n_speakers=n_speakers
        )

        self.sep_start = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=enc_hidden_dim),
            nn.Conv1d(
                in_channels=enc_hidden_dim, out_channels=enc_hidden_dim, kernel_size=1
            ),
        )

        self.separation = DPRNN(
            n_frames=n_frames,
            input_dim=enc_hidden_dim,
            hidden_dim=rnn_hidden_dim,
            n_dprnn_blocks=n_dprnn_blocks,
            overlap=overlap,
        )

        self.sep_final = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(
                in_channels=enc_hidden_dim,
                out_channels=enc_hidden_dim * n_speakers,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

        self.decoder = Decoder(input_dim=enc_hidden_dim, hidden_dim=input_dim)

    def forward(self, audio_mix, **batch):
        residual, x = self.encoder(audio_mix)

        x = self.sep_start(x)

        x = self.separation(x)

        x = self.sep_final(x) * residual

        x = x.contiguous().view(-1, self.n_speakers, self.enc_hidden_dim, self.n_frames)

        return {
            "audio_pred_first": self.decoder(x[:, 0]),
            "audio_pred_second": self.decoder(x[:, 1]),
        }
