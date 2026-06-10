
from numpy import real
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import typing as tp
from torch.nn.parameter import Parameter
from torch.nn import init
from models.get_layer_from_string import get_layer
from models.stftv2 import STFT, iSTFT
from models.local.dcunet import Encoder, Decoder

EPS = 1e-8

def select_norm(norm, dim, shape, eps=1e-8):
    """Just a wrapper to select the normalization type.
    """
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=eps)
    else:
        return nn.BatchNorm1d(dim)

def get_fftfreq(
        sr: int = 16000,
        n_fft: int = 512
) -> torch.Tensor:
    """
    Torch workaround of librosa.fft_frequencies
    """
    out = sr * torch.fft.fftfreq(n_fft)[:n_fft // 2 + 1]
    out[-1] = sr // 2
    return out


def get_subband_indices(
        freqs: torch.Tensor,
        splits: tp.List[tp.Tuple[int, int]],
) -> tp.List[tp.Tuple[int, int]]:
    """
    Computes subband frequency indices with given bandsplits
    """
    indices = []
    start_freq, start_index = 0, 0
    for end_freq, step in splits:
        bands = torch.arange(start_freq + step, end_freq + step, step)
        start_freq = end_freq
        for band in bands:
            end_index = freqs[freqs < band].shape[0]
            indices.append((start_index, end_index))
            start_index = end_index
    indices.append((start_index, freqs.shape[0]))
    return indices


def freq2bands(
        bandsplits: tp.List[tp.Tuple[int, int]],
        sr: int = 16000,
        n_fft: int = 512
    ) -> tp.List[tp.Tuple[int, int]]:
        """
        Returns start and end FFT indices of given bandsplits
        """
        freqs = get_fftfreq(sr=sr, n_fft=n_fft)
        band_indices = get_subband_indices(freqs, bandsplits)
        return band_indices

class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat

class AllHeadPReLULayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 3
        H, E, n_freqs = input_dimension
        param_size = [1, H, E, 1, n_freqs]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.act = nn.PReLU(num_parameters=H, init=0.25)
        self.eps = eps
        self.H = H
        self.E = E
        self.n_freqs = n_freqs

    def forward(self, x):
        assert x.ndim == 4
        B, _, T, _ = x.shape
        x = x.view([B, self.H, self.E, T, self.n_freqs])
        x = self.act(x)  # [B,H,E,T,F]
        stat_dim = (2, 4)
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,H,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,H,1,T,1]
        x = ((x - mu_) / std_) * self.gamma + self.beta  # [B,H,E,T,F]
        return x

class GLU(nn.Module):
    """
    GLU Activation Module.
    """
    def __init__(self, input_dim: int):
        super(GLU, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, input_dim * 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = x[..., :self.input_dim] * self.sigmoid(x[..., self.input_dim:])
        return x


class MLP(nn.Module):
    """
    Just a simple MLP with tanh activation (by default).
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            activation_type: str = 'tanh',
    ):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.select_activation(activation_type)(),
            nn.Linear(hidden_dim, output_dim),
            GLU(output_dim)
        )

    @staticmethod
    def select_activation(activation_type: str) -> nn.modules.activation:
        if activation_type == 'tanh':
            return nn.Tanh
        elif activation_type == 'relu':
            return nn.ReLU
        elif activation_type == 'gelu':
            return nn.GELU
        else:
            raise ValueError("wrong activation function was selected")

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        return x

class GridNetV2Block(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        # spk_emb_dim,
        hidden_channels,
        eps=1e-5,
    ):
        super().__init__()

        # in_channels = emb_dim * emb_ks + spk_emb_dim
        in_channels = emb_dim * emb_ks

        # self.intra_l1 = nn.Linear(spk_emb_dim, emb_dim)
        # self.intra_l2 = nn.Linear(spk_emb_dim, emb_dim)

        # self.inter_l1 = nn.Linear(spk_emb_dim, emb_dim)
        # self.inter_l2 = nn.Linear(spk_emb_dim, emb_dim)

        self.intra_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        if emb_ks == emb_hs:
            self.intra_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.intra_linear = nn.ConvTranspose1d(
                hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
            )

        self.inter_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=False
        )
        if emb_ks == emb_hs:
            self.inter_linear = nn.Linear(hidden_channels, in_channels)
        else:
            self.inter_linear = nn.ConvTranspose1d(
                hidden_channels, emb_dim, emb_ks, stride=emb_hs
            )

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs

    def forward(self, x):
        """GridNetV2Block Forward.

        Args:
            x: [B, C, T, Q]
            out: [B, C, T, Q]
        """
        B, C, old_T, old_Q = x.shape

        olp = self.emb_ks - self.emb_hs
        T = (
            math.ceil((old_T + 2 * olp - self.emb_ks) / self.emb_hs) * self.emb_hs
            + self.emb_ks
        )
        Q = (
            math.ceil((old_Q + 2 * olp - self.emb_ks) / self.emb_hs) * self.emb_hs
            + self.emb_ks
        )

        x = x.permute(0, 2, 3, 1)  # [B, old_T, old_Q, C]
        x = F.pad(x, (0, 0, olp, Q - old_Q - olp, olp, T - old_T - olp))  # [B, T, Q, C]

        # intra RNN
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, T, Q, C]
        # embd_intra = emb.unsqueeze(1).repeat(1,input_.shape[1],input_.shape[2],1)
        # intra_rnn = self.intra_norm(self.intra_l1(embd_intra)*input_ + self.intra_l2(embd_intra))
        if self.emb_ks == self.emb_hs:
            intra_rnn = intra_rnn.view([B * T, -1, self.emb_ks * C])  # [BT, Q//I, I*C]
            # embd_intra = emb.unsqueeze(1).repeat(1,T,intra_rnn.shape[1],1).view(B*T, intra_rnn.shape[1], emb.shape[-1])
            # intra_rnn, _ = self.intra_rnn(torch.cat([intra_rnn,embd_intra], dim=-1))  # [BT, Q//I, H]
            intra_rnn, _ = self.intra_rnn(intra_rnn)
            intra_rnn = self.intra_linear(intra_rnn)  # [BT, Q//I, I*C]
            intra_rnn = intra_rnn.view([B, T, Q, C])
        else:
            intra_rnn = intra_rnn.view([B * T, Q, C])  # [BT, Q, C]
            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, C, Q]
            intra_rnn = F.unfold(
                intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
            )  # [BT, C*I, -1]
            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*I]
            # embd_intra = emb.unsqueeze(1).repeat(1,T,intra_rnn.shape[1],1).view(B*T, intra_rnn.shape[1], emb.shape[-1])

            # intra_rnn, _ = self.intra_rnn(torch.cat([intra_rnn,embd_intra], dim=-1))  # [BT, -1, H]

            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
            intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
            intra_rnn = intra_rnn.view([B, T, C, Q])
            intra_rnn = intra_rnn.transpose(-2, -1)  # [B, T, Q, C]
        intra_rnn = intra_rnn + input_  # [B, T, Q, C]

        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, Q, T, C]

        # inter RNN
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)  # [B, Q, T, C]
        # embd_inter = emb.unsqueeze(1).repeat(1,input_.shape[1],input_.shape[2],1)
        # inter_rnn = self.inter_norm(self.inter_l1(embd_inter)*input_ + self.inter_l2(embd_inter))
        if self.emb_ks == self.emb_hs:
            inter_rnn = inter_rnn.view([B * Q, -1, self.emb_ks * C])  # [BQ, T//I, I*C]
            # embd_inter = emb.unsqueeze(1).repeat(1,Q,inter_rnn.shape[1],1).view(B*Q, inter_rnn.shape[1], emb.shape[-1])
            # inter_rnn, _ = self.inter_rnn(torch.cat([inter_rnn,embd_inter], dim=-1))  # [BQ, T//I, H]
            inter_rnn, _ = self.inter_rnn(inter_rnn)
            inter_rnn = self.inter_linear(inter_rnn)  # [BQ, T//I, I*C]
            inter_rnn = inter_rnn.view([B, Q, T, C])
        else:
            inter_rnn = inter_rnn.view(B * Q, T, C)  # [BQ, T, C]
            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, C, T]
            inter_rnn = F.unfold(
                inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
            )  # [BQ, C*I, -1]
            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, -1, C*I]
            # embd_inter = emb.unsqueeze(1).repeat(1,Q,inter_rnn.shape[1],1).view(B*Q, inter_rnn.shape[1], emb.shape[-1])

            # inter_rnn, _ = self.inter_rnn(torch.cat([inter_rnn,embd_inter], dim=-1))  # [BQ, -1, H]
            inter_rnn, _ = self.inter_rnn(inter_rnn)

            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, H, -1]
            inter_rnn = self.inter_linear(inter_rnn)  # [BQ, C, T]
            inter_rnn = inter_rnn.view([B, Q, C, T])
            inter_rnn = inter_rnn.transpose(-2, -1)  # [B, Q, T, C]
        inter_rnn = inter_rnn + input_  # [B, Q, T, C]

        inter_rnn = inter_rnn.permute(0, 3, 2, 1)  # [B, C, T, Q]

        inter_rnn = inter_rnn[..., olp : olp + old_T, olp : olp + old_Q]
        
        return inter_rnn

class Tar_Model(nn.Module):

    def __init__(
        self,
        n_fft,
        hop_length,
        win_length,
        embd_dim,
        feature_dim,
        fc_dim,
        emb_ks,
        emb_hs,
        # spk_emb_dim,
        num_layers,
        lstm_hidden_units,
        eps=1.0e-5,
        
    ):
        super(Tar_Model, self).__init__()

        self.num_layers = num_layers

        self.stft = STFT(n_fft=n_fft,
                         hop_length=hop_length,
                         win_length=win_length)
        self.istft = iSTFT(n_fft=n_fft,
                         hop_length=hop_length,
                         win_length=win_length)

        self.l1 = nn.Linear(embd_dim, feature_dim)
        self.l2 = nn.Linear(feature_dim, 129)
        self.l3 = nn.Linear(129, 65)
        self.l4 = nn.Linear(65, 33)
        self.l5 = nn.Linear(33, 17)


        self.fgru = nn.LSTM(41, 40, num_layers=1, batch_first=True, bidirectional=True)
        self.flinear = nn.Linear(80,40)
        # self.tgru = nn.GRU(17*40+256, 17*40, num_layers=2, batch_first=True, bidirectional=False)
        # self.tlinear = nn.Linear(17*40,17*40) #
        self.tgru = nn.LSTM(41, 80, num_layers=2, batch_first=True, bidirectional=False)
        self.tlinear = nn.Linear(80,40)

        self.downsample0 = Encoder(filter_size=(5,1), stride_size=(2,1), in_channels=2, out_channels=20, padding=(2,0))
        # self.downsample1 = Encoder(filter_size=(7,1), stride_size=(1,1), embd_dim = 256, in_channels=45, out_channels=90)
        self.downsample1 = Encoder(filter_size=(5,1), stride_size=(2,1), in_channels=20, out_channels=20, padding=(2,0))
        self.downsample2 = Encoder(filter_size=(5,1), stride_size=(2,1), in_channels=20, out_channels=20, padding=(2,0))
        self.downsample3 = Encoder(filter_size=(5,1), stride_size=(2,1), in_channels=20, out_channels=40, padding=(2,0))
        
        # upsampling/decoding
        self.upsample0 = Decoder(filter_size=(5,1), stride_size=(2,1), in_channels=40, out_channels=20, padding=(2,0))
        self.upsample1 = Decoder(filter_size=(5,1), stride_size=(2,1), in_channels=20, out_channels=20, padding=(2,0))
        self.upsample2 = Decoder(filter_size=(5,1), stride_size=(2,1), in_channels=20, out_channels=20, padding=(2,0))
        # self.upsample3 = Decoder(filter_size=(7,1), stride_size=(1,1), in_channels=180, out_channels=45)
        self.upsample3 = Decoder(filter_size=(5,1), stride_size=(2,1), in_channels=20, out_channels=1, padding=(2,0))

        # self.l1 = nn.Linear(embd_dim, fc_dim)

        t_ksize = 1
        ks, padding = (t_ksize, 1), (t_ksize // 2, 0)
        # ks, padding = (1, 3), (0, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, fc_dim, ks, padding=padding),
            nn.GroupNorm(1, fc_dim, eps=eps),
        )

        # self.attn = nn.MultiheadAttention(fc_dim,
        #                                   num_heads=4,
        #                                   dropout=0,
        #                                   batch_first=True)

        self.deconv = nn.ConvTranspose2d(fc_dim, 2, ks, padding=padding)
        self.ac = nn.ReLU()

        # self.alpha = Parameter(torch.Tensor(*[1,1,fc_dim]).to(torch.float32))

        # self.fusion_conv = nn.Conv2d(fc_dim+embd_dim, fc_dim, 1, 1)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fc_dim+embd_dim, fc_dim, ks, padding=padding),
            nn.GroupNorm(1, fc_dim, eps=eps),
        )

        self.dual_mdl = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    GridNetV2Block(
                        fc_dim,
                        emb_ks,
                        emb_hs,
                        # spk_emb_dim,
                        lstm_hidden_units,
                        eps,
                    )
                )
            )
    
        
    def forward(self, input, aux):
        # print(input.shape, aux.shape) # [B, L], [B, T, E]

        input = input.unsqueeze(1) # [B, 1, L]
        std = input.std(dim=(1, 2), keepdim=True)
        input = input / (std+EPS)
        
        input_c = self.stft(input)[-1]        
        x_w = torch.cat([input_c.real, input_c.imag], dim=1) # [B, 2, F, T]
        xs_mag = torch.abs(x_w)**0.5 # [B, 2, F, T]
        xs_phase = torch.angle(x_w) # [B, 2, F, T]
        # xs = torch.cat([input_c.real, input_c.imag], dim=1)  #B,4,F,T

        aux = torch.nn.functional.interpolate(aux.transpose(1, 2), size=(x_w.shape[-1]), mode='linear') # [B, E, T]
        # aux = aux.transpose(1, 2) # [B, T, E]
        emb_mag_ori = self.l1(aux.transpose(1, 2)) # [B, T, F]
        emb_mag_ori = emb_mag_ori.unsqueeze(1).transpose(2, 3) # [B, 1, F, T]
        # emb_mag_ori = emb_mag_ori.unsqueeze(-1).repeat(1,1,1,xs_mag.shape[-1]) # [B, T, F, T]

        emb_mag = emb_mag_ori

        # print(torch.cat([xs_mag, emb_mag],dim=1).shape)

        # print(xs_mag.shape, emb_mag_ori.shape)

        # xs = self.conv(torch.cat([xs_mag, emb_mag], dim = 1)) #B,fc_dim+1,F,T

        # downsampling/encoding
        d0 = self.downsample0(torch.cat([xs_mag, emb_mag], dim = 1))#B,C,F,T
        # print('d0', d0.shape)
        emb_mag = self.l2(emb_mag.squeeze(1).permute(0,2,1).contiguous())#B,F,T
        emb_mag = emb_mag.permute(0,2,1).contiguous().unsqueeze(1)#B,1,F,T
        # d0 = torch.cat([d0, emb], dim = 1)

        d1 = self.downsample1(torch.cat([d0, emb_mag], dim = 1))
        emb_mag = self.l3(emb_mag.squeeze(1).permute(0,2,1).contiguous())
        emb_mag = emb_mag.permute(0,2,1).contiguous().unsqueeze(1)
        # d1 = torch.cat([d1, emb], dim = 1)
        # print(d1.shape, emb.shape) 
        d2 = self.downsample2(torch.cat([d1, emb_mag], dim = 1))
        emb_mag = self.l4(emb_mag.squeeze(1).permute(0,2,1).contiguous())
        emb_mag = emb_mag.permute(0,2,1).contiguous().unsqueeze(1)
        # d2 = torch.cat([d2, emb], dim = 1)        
        d3 = self.downsample3(torch.cat([d2, emb_mag], dim = 1))
        # print(d3.shape)        
        # d4 = self.downsample4(d2)
        emb_mag = self.l5(emb_mag.squeeze(1).permute(0,2,1).contiguous())
        emb_mag = emb_mag.permute(0,2,1).contiguous().unsqueeze(1) #B,1,F,T

        # print(emb.shape)
        
        # emb = aux.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,x_w.shape[-2],2)
        d3_bottleneckF = torch.cat([d3,emb_mag],dim=1)
        B,C,F,T = d3_bottleneckF.shape
        # d3_bottleneckF = torch.cat([d3,emb],dim=1) ###
        # d3_bottleneckF = d3
        # B,C,F,T = d3_bottleneckF.shape
        d3_bottleneckF = d3_bottleneckF.permute(0,3,2,1).contiguous().view(B*T, F, C)
        d3_bottleneckF = self.fgru(d3_bottleneckF)[0]
        d3_bottleneckF = self.flinear(d3_bottleneckF)
        d3_bottleneckF = d3_bottleneckF.view(B,T,F,C-1).permute(0,3,2,1).contiguous() #B,C-1,F,T
        d3_bottleneckF = d3_bottleneckF + d3

        d3_bottleneckT = torch.cat([d3_bottleneckF,emb_mag],dim=1)
        # emb_t = aux.unsqueeze(-1).repeat(1,C-1,F,T)ff
        # d3_bottleneckT = torch.cat([d3_bottleneckF,emb_t],dim=-2)
        
        # B*F, T, C-1
        d3_bottleneckT = d3_bottleneckT.permute(0,2,3,1).contiguous().view(B*F, T, C) ###
        # emb_t = aux.repeat(B,T,1)
        # d3_bottleneckT = torch.cat([d3_bottleneckT,emb_t],dim=-1)
        # print(d3_bottleneckT.shape, emb_t.shape)
        d3_bottleneckT = self.tgru(d3_bottleneckT)[0]
        d3_bottleneckT = self.tlinear(d3_bottleneckT)
        d3_bottleneckT = d3_bottleneckT.contiguous().view(B,T,F,C-1).permute(0,3,2,1).contiguous()

        d3_bottleneck = d3_bottleneckT + d3_bottleneckF

        # upsampling/decoding 
        u0 = self.upsample0(d3_bottleneck)
        # print(u0.shape,d3_bottleneck.shape,d2.shape)
        c0 = u0 + d2
        
        u1 = self.upsample1(c0)
        c1 = u1 + d1

        u2 = self.upsample2(c1)
        c2 = u2 + d0

        u3 = self.upsample3(c2)

        # u3 = self.ac(u3)
        # u3 = u3*xs_mag[:,0,:,:].unsqueeze(1)
        # print(u3.shape)


        real_part = u3 * torch.cos(xs_phase[:,0,:,:].unsqueeze(1)) + xs_mag[:,0,:,:].unsqueeze(1) * torch.cos(xs_phase[:,0,:,:].unsqueeze(1))
        imag_part = u3 * torch.sin(xs_phase[:,0,:,:].unsqueeze(1)) + xs_mag[:,0,:,:].unsqueeze(1) * torch.sin(xs_phase[:,0,:,:].unsqueeze(1))
        # print(real_part.shape, imag_part.shape)


        xs = torch.cat([real_part, imag_part], dim = 1)

        xs = self.conv(xs) # [B, C, F, T]

        # emb_ri = emb_mag_ori # [B, E, T] -> [B, E, F, T]
        emb_ri= aux.unsqueeze(2).repeat(1,1,xs.shape[2],1) # [B, E, F, T]

        # emb = emb.unsqueeze(-1).repeat(1,1,1,xs.shape[-1])
        xs = torch.cat([xs, emb_ri], dim=1)

        # print(emb_ri.shape, xs.shape)

        xs = self.fusion_conv(xs)

        # B,C,T,F
        xs = xs.transpose(2,3)

        # xs = self.conv(xs) #B,_,T,F

        for i in range(self.num_layers):
            xs = self.dual_mdl[i](xs)
        
        outs = self.deconv(xs) # B,2,T,F
        # print(outs.shape)
        outs_R = outs[:,0,:,:].permute(0,2,1).contiguous() + real_part.squeeze(1)
        outs_I = outs[:,1,:,:].permute(0,2,1).contiguous() + imag_part.squeeze(1) #B,F,T
        # print(outs_R.shape)
        scale_p = (outs_R ** 2 + outs_I ** 2) / ((outs_R ** 2 + outs_I ** 2) ** 0.5 + EPS)
        # out_put_phase = torch.atan2(outs_I, outs_R+EPS)
        outs_R = outs_R * scale_p
        outs_I = outs_I * scale_p

        # out_put = self.istft((out_put_mag, out_put_phase), input_type="mag_phase")

        out_put = self.istft((outs_R, outs_I), input_type="real_imag") # B,1,T
        # print(out_put.shape)
        out_put = out_put.unsqueeze(1) * std

        return out_put.squeeze(1)




