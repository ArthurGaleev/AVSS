import torch
from torch import nn
import torch.nn.functional as F
import math


class OneDConv(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.dconv1d_block = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1),

            nn.PReLU(),
            nn.GroupNorm(num_groups=1, num_channels=hidden_dim), # TODO: add cumulative layer norm, because the task is causal

            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),

            nn.PReLU(),
            nn.GroupNorm(num_groups=1, num_channels=hidden_dim), # TODO: add cumulative layer norm, because the task is causal
            )
        
        self.res_conv = nn.Conv1d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=1)
        self.out_conv = nn.Conv1d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=1)
    
    def forward(self, audio_mix):
        x = self.dconv1d_block(audio_mix.unsqueeze(1))  # expected num_channels=1 in conv1d

        residual = self.res_conv(x)
        out = self.out_conv(x) + audio_mix.unsqueeze(1)
        
        return residual, out
    

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.dconv1d_block = nn.Sequential(
            nn.ConvTranspose1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1),

            nn.PReLU(),
            nn.GroupNorm(num_groups=1, num_channels=hidden_dim), # TODO: add cumulative layer norm, because the task is causal

            nn.ConvTranspose1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),

            nn.PReLU(),
            nn.GroupNorm(num_groups=1, num_channels=hidden_dim), # TODO: add cumulative layer norm, because the task is causal
            )

        self.out_conv = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
    
    def forward(self, x):
        out = self.dconv1d_block(x)

        out = self.out_conv(out) + x
        
        return out


class Segmentation(nn.Module):
    def __init__(self, n_frames, overlap=True):
        super().__init__()
        self.n_frames = n_frames
        self.overlap = overlap
    
    def forward(self, x):
        # as in original DPRNN paper
        # chunk_size = √(2*n_frames) = 2*p
        p = int(math.sqrt(2*self.n_frames) // 2)

        if self.overlap:
            p_shift = p  # 50% overlap
        else:
            p_shift = 2*p # no overlap

        chunks = []
        chunks.append(F.pad(x[:, :, :p], (0, p)))  # add the first sample padded to chunk_size
        
        for i in range(p_shift, p * (self.n_frames // p - 1), p_shift):
            chunks.append(x[:, :, i:(i+2*p)])  # add the next sample of chunk_size

        chunks.append(F.pad(x[:, :, (i+p):], (0, p - (self.n_frames % p)))) # add the last sample padded to chunk_size
        
        # need to permute stacked chunks of shape
        # (n_chunks x batch_size x n_features x chunk_size) -> (batch_size x n_features x chunk_size x n_chunks)
        return torch.stack(chunks).permute(1, 2, 3, 0)
    

class OverlapAdd(nn.Module):
    def __init__(self, n_frames, overlap=True):
        super().__init__()
        self.n_frames = n_frames
        self.overlap = overlap

    def forward(self, x):
        batch_size, n_features, chunk_size, n_chunks = x.shape

        out = torch.zeros((batch_size, n_features, self.n_frames))

        # as in original DPRNN paper
        # chunk_size = √(2*n_frames) = 2*p
        p = int(math.sqrt(2*self.n_frames) // 2)

        overlap = True
        if overlap:
            p_shift = p  # 50% overlap
        else:
            p_shift = 2*p # no overlap

        for i in range(0, n_chunks-1):
            chunk_start = i * p_shift
            out[:, :, chunk_start:(chunk_start+2*p)] += x[:, :, :, i].squeeze(-1)

        # last chunk we cut a little bit to fit the out shape
        chunk_start += p_shift
        out[:, :, chunk_start:(chunk_start+2*p)] += x[:, :, :(self.n_frames % (2*p)), i].squeeze(-1)
        
        return out
    

class DPRNNBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.intra_rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.intra_fc = nn.Linear(in_features=2*hidden_dim, out_features=input_dim)
        self.intra_norm = nn.LayerNorm(input_dim)

        self.inter_rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.inter_fc = nn.Linear(in_features=2*hidden_dim, out_features=input_dim)
        self.inter_norm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        batch_size, n_features, chunk_size, n_chunks = x.shape

        # because intra_rnn is applied to individual chunks
        x = x.permute(0, 3, 2, 1).contiguous().view(batch_size*n_chunks, chunk_size, n_features)
        x = self.intra_norm(self.intra_fc(self.intra_rnn(x)[0])) + x
        
        # because inter_rnn is applied across the chunks
        x = x.view(batch_size*chunk_size, n_chunks, n_features)
        x = self.inter_norm(self.inter_fc(self.inter_rnn(x)[0])) + x
        
        # reshape back
        return x.view(batch_size, n_features, chunk_size, n_chunks)
    

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
        
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=enc_hidden_dim)

        self.sep_start = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=input_dim),
            nn.Conv1d(in_channels=input_dim, out_channels=enc_hidden_dim, kernel_size=1),
        )

        self.separation = DPRNN(
            n_frames=n_frames,
            input_dim=enc_hidden_dim, 
            hidden_dim=rnn_hidden_dim,
            n_dprnn_blocks=n_dprnn_blocks,
            overlap=overlap
        )

        self.sep_final = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(in_channels=enc_hidden_dim, out_channels=enc_hidden_dim*n_speakers, kernel_size=1),
            nn.Sigmoid(),
        )
        
        self.decoder = Decoder(input_dim=enc_hidden_dim, hidden_dim=input_dim)


    def forward(self, audio_mix, **batch):
        residual, x = self.encoder(audio_mix)
        
        x = self.sep_start(x)

        x = self.separation(x)*residual

        x = self.sep_final(x).view(-1,  self.n_speakers, self.enc_hidden_dim, self.n_frames)

        return {
            "audio_pred_first": self.decoder(x[:, 0]),
            "audio_pred_second": self.decoder(x[:, 1])
        }


