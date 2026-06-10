
import torch
import torch.nn as nn
# import torch.nn.functional as F
import copy
import math
import typing as tp
from torch.nn.parameter import Parameter
from torch.nn import init
from models.local.dcunet import Encoder, Decoder
from models.stftv2 import STFT, iSTFT
# import torch.nn.functional as F

EPS = 1e-8

class Tar_Model(nn.Module):

    def __init__(
        self,
        n_fft,
        hop_length,
        win_length,
        embd_dim,
        feature_dim,
        # n_fft,
        # hop_length,
        eps=1.0e-5,
        
    ):
        super(Tar_Model, self).__init__()

        # self.n_fft = n_fft
        # self.hop_length = hop_length

        self.stft = STFT(n_fft=n_fft,
                         hop_length=hop_length,
                         win_length=win_length)
        self.istft = iSTFT(n_fft=n_fft,
                         hop_length=hop_length,
                         win_length=win_length)

        self.embd_dim = embd_dim

        self.l1 = nn.Linear(embd_dim, feature_dim)
        self.l2 = nn.Linear(feature_dim, 129)
        self.l3 = nn.Linear(129, 65)
        self.l4 = nn.Linear(65, 33)
        self.l5 = nn.Linear(33, 17)


        self.fgru = nn.GRU(21, 40, num_layers=1, batch_first=True, bidirectional=True)
        self.flinear = nn.Linear(80,20)
        self.tgru = nn.GRU(340+embd_dim, 340, num_layers=2, batch_first=True, bidirectional=False)
        self.tlinear = nn.Linear(340,340) #

        self.downsample0 = Encoder(filter_size=(5,1), stride_size=(2,1), in_channels=2, out_channels=16, padding=(2,0))
        # self.downsample1 = Encoder(filter_size=(7,1), stride_size=(1,1), embd_dim = 256, in_channels=45, out_channels=90)
        self.downsample1 = Encoder(filter_size=(5,1), stride_size=(2,1), in_channels=16, out_channels=16, padding=(2,0))
        self.downsample2 = Encoder(filter_size=(5,1), stride_size=(2,1), in_channels=16, out_channels=16, padding=(2,0))
        self.downsample3 = Encoder(filter_size=(5,1), stride_size=(2,1), in_channels=16, out_channels=20, padding=(2,0))
        
        # upsampling/decoding
        self.upsample0 = Decoder(filter_size=(5,1), stride_size=(2,1), in_channels=20, out_channels=16, padding=(2,0))
        self.upsample1 = Decoder(filter_size=(5,1), stride_size=(2,1), in_channels=16, out_channels=16, padding=(2,0))
        self.upsample2 = Decoder(filter_size=(5,1), stride_size=(2,1), in_channels=16, out_channels=16, padding=(2,0))
        # self.upsample3 = Decoder(filter_size=(7,1), stride_size=(1,1), in_channels=180, out_channels=45)
        self.upsample3 = Decoder(filter_size=(5,1), stride_size=(2,1), in_channels=16, out_channels=2, padding=(2,0))

        
    def forward(self, input, aux):
        # print(input.shape, aux.shape)
        # aux.shape = 2,75,512
        # input = input.unsqueeze(1) #2,1,48000
        # input_c = torch.stft(input=input.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, normalized=True, return_complex=False) #B,F,T,2
        std = input.std(dim=(1, 2), keepdim=True)
        input = input / (std+EPS)
        input_c = self.stft(input)[-1]
        
        x_w = torch.cat([input_c.real, input_c.imag], dim=1) #2, 2, 257, 151
        # print(x_w.shape)
        
        aux = torch.nn.functional.interpolate(aux.transpose(1, 2), size=(x_w.shape[-1]), mode='linear') # [B, E, T]
        aux = aux.transpose(1, 2) # [B, T, E]
        emb = self.l1(aux) # [B, T, F]
        
        emb = emb.unsqueeze(1).transpose(2, 3) # [B, 1, F, T]
        # print(emb.shape)
        x = torch.cat([x_w, emb], dim=1) # [B, 3, F, T]
        # print(emb.shape)
        
        # downsampling/encoding
        d0 = self.downsample0(x)
        # print(d0.shape)
        emb = self.l2(emb.squeeze(1).permute(0,2,1).contiguous())
        emb = emb.permute(0,2,1).contiguous().unsqueeze(1)
        # d0 = torch.cat([d0, emb], dim = 1)

        d1 = self.downsample1(torch.cat([d0, emb], dim = 1))
        emb = self.l3(emb.squeeze(1).permute(0,2,1).contiguous())
        emb = emb.permute(0,2,1).contiguous().unsqueeze(1)
        # d1 = torch.cat([d1, emb], dim = 1)
        # print(d1.shape, emb.shape) 
        d2 = self.downsample2(torch.cat([d1, emb], dim = 1))
        emb = self.l4(emb.squeeze(1).permute(0,2,1).contiguous())
        emb = emb.permute(0,2,1).contiguous().unsqueeze(1)
        # d2 = torch.cat([d2, emb], dim = 1)        
        d3 = self.downsample3(torch.cat([d2, emb], dim = 1))
        # print(d3.shape)        
        # d4 = self.downsample4(d2)
        emb = self.l5(emb.squeeze(1).permute(0,2,1).contiguous())
        emb = emb.permute(0,2,1).contiguous().unsqueeze(1)
        
        # emb = aux.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,x_w.shape[-2],2)
        d3_bottleneckF = torch.cat([d3,emb],dim=1)
        B,C,F,T = d3_bottleneckF.shape
        # d3_bottleneckF = torch.cat([d3,emb],dim=1) ###
        # d3_bottleneckF = d3
        # B,C,F,T = d3_bottleneckF.shape
        d3_bottleneckF = d3_bottleneckF.permute(0,3,2,1).contiguous().view(B*T, F, C)
        d3_bottleneckF = self.fgru(d3_bottleneckF)[0]
        d3_bottleneckF = self.flinear(d3_bottleneckF)
        d3_bottleneckF = d3_bottleneckF.view(B,T,F,C-1).permute(0,3,2,1).contiguous()
        d3_bottleneckF = d3_bottleneckF + d3

        # d3_bottleneckT = torch.cat([d3_bottleneckF,emb],dim=1)
        # emb_t = aux.unsqueeze(-1).repeat(1,C-1,F,T)ff
        # d3_bottleneckT = torch.cat([d3_bottleneckF,emb_t],dim=-2)
        d3_bottleneckT = d3_bottleneckF.permute(0,3,2,1).contiguous().view(B, T, F*(C-1)) ###
        emb_t = aux
        # emb_t = aux.repeat(1,T,1)
        d3_bottleneckT = torch.cat([d3_bottleneckT,emb_t],dim=-1)
        d3_bottleneckT = self.tgru(d3_bottleneckT)[0]
        d3_bottleneckT = self.tlinear(d3_bottleneckT)
        d3_bottleneckT = d3_bottleneckT.view(B,T,F,C-1).permute(0,3,2,1).contiguous()

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

        # print(u4.shape)
        out_put = u3 * x_w
        # print(out_put.shape)

        # out_put = out_put * std

        # out_put = u3

        # out_put = torch.cat([out_put, input_c[:,:,0,:]], dim=2)
        # print(out_put.shape)

        # out_put = torch.istft(out_put, n_fft=self.n_fft, hop_length=self.hop_length, normalized=True)
        # out_put_R = torch.cat([input_c.real[:,:,0,:], out_put[:,0,:,:]], dim=2)
        # out_put_I = torch.cat([input_c.imag[:,:,0,:], out_put[:,1,:,:]], dim=2)
        out_put_R = out_put[:,0,:,:]
        out_put_I = out_put[:,1,:,:]

        # out_put_mag = out_put_R ** 2 + out_put_I ** 2
        # out_put_phase = torch.atan2(out_put_I, out_put_R+EPS)

        # out_put = self.istft((out_put_mag, out_put_phase), input_type="mag_phase")

        out_put = self.istft((out_put_R, out_put_I), input_type="real_imag")

        out_put = out_put.unsqueeze(1) * std
        
        # print(out_put.shape)

        return out_put.squeeze(1)




