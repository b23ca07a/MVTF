import sys
import os

import scipy, math
import numpy as np
import torch
import yaml
import time
from utils.genrir import RandomRirGenerator
from scipy import io
EPS = 1e-8

with open('config/std_train.yml') as rfile:
    config = yaml.safe_load(rfile)
rirgen = RandomRirGenerator(**config['RandomRirGenerator_kwargs'])

def get_instance(module, config, *args, **kwargs):
    return getattr(module, config['type'])(*args, **kwargs, **config['args'])

SPEECH_FILTER = io.loadmat(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "speech_weight.mat"
    ),
    squeeze_me=True,
)
# SPEECH_FILTER = np.array(SPEECH_FILTER["filt"])
SPEECH_FILTER = torch.from_numpy(SPEECH_FILTER["filt"]).float()
# SPEECH_FILTER = np.array(SPEECH_FILTER["filt"])
def apply_ramp(y, fs=16000, dur=0.5):
    """Apply half cosine ramp into and out of signal

    dur - ramp duration in seconds
    """
    ramp = np.cos(np.linspace(math.pi, 2 * math.pi, int(fs * dur)))
    ramp = (ramp + 1) / 2
    # y = np.array(x)
    y[0 : len(ramp)] *= ramp
    y[-len(ramp) :] *= ramp[::-1]
    return y

# def apply_ramp(y, ilens, fs=16000, dur=0.5):
#     """
#     y: [B, T]
#     """
#     ramp = torch.cos(torch.linspace(math.pi, 2 * math.pi, int(fs * dur), device=y.device))
#     ramp = (ramp + 1) / 2
#     y[:,:len(ramp)] *= ramp
#     for i in range(y.shape[0]):
#         y[i,(ilens[i]-len(ramp)):ilens[i]] *= ramp.flip([0])
#     return y

def speechweighted_snr(target, noise):
    """Apply speech weighting filter to signals and get SNR."""
    target_filt = scipy.signal.convolve(
        target, SPEECH_FILTER, mode="full", method="fft"
    )
    noise_filt = scipy.signal.convolve(noise, SPEECH_FILTER, mode="full", method="fft")

    # rms of the target after speech weighted filter
    targ_rms = np.sqrt(np.mean(target_filt ** 2))

    # rms of the noise after speech weighted filter
    noise_rms = np.sqrt(np.mean(noise_filt ** 2))

    if noise_rms==0:
        print('tar', targ_rms, 'inter', noise_rms, 'max amp', np.max(np.abs(noise)), flush=True)
        return 0
    
    sw_snr = np.divide(targ_rms, noise_rms)
    return sw_snr

# def speechweighted_snr(speech, filter):
#     """
#     Apply speech weighting filter to signals and get SNR.
#     input: [B, 2, T], 0:target, 1:noise
#     output: [B]
#     """
#     speech = torch.nn.functional.pad(speech, (filter.shape[-1] - 1,filter.shape[-1] - 1), 'constant', 0) # [B, 2, T+2(F-1)]
#     speech_filt = torch.nn.functional.conv1d(speech, filter, groups=2) # [B, 2, T]
#     speech_rms = torch.sqrt(torch.mean(speech_filt ** 2, dim=2)) # [B, 2]

#     # if noise_rms==0:
#     #     print('tar', targ_rms, 'inter', noise_rms, 'max amp', torch.max(torch.abs(noise)), flush=True)
#     #     return 0
    
#     sw_snr = torch.divide(speech_rms[:,0], speech_rms[:,1]) # [B]
#     return sw_snr

def mixing(inter_wav, tar_wav, snr_dB):
    """
    inter_wav: [T]
    snr_dB: float
    output: [T]
    """
    inter_wav = apply_ramp(inter_wav)
    snr_ref = speechweighted_snr(tar_wav, inter_wav)
    inter_wav = inter_wav * (snr_ref + EPS)
    inter_wav = inter_wav * 10 ** ((-snr_dB) / 20)
    mix_wav = inter_wav + tar_wav
    all_signals = np.concatenate((mix_wav, tar_wav, inter_wav))
    norm = np.max(np.abs(all_signals))
    mix_wav /= norm
    tar_wav /= norm
    return mix_wav, tar_wav

#================================================================================================
# def apply_ramp(y, fs=16000, dur=0.5):
#     """Apply half cosine ramp into and out of signal

#     dur - ramp duration in seconds
#     """
#     ramp = np.cos(np.linspace(math.pi, 2 * math.pi, int(fs * dur)))
#     ramp = (ramp + 1) / 2
#     # y = np.array(x)
#     y[0 : len(ramp)] *= ramp
#     y[-len(ramp) :] *= ramp[::-1]
#     return y


# def apply_ramp(y, ilens, fs=16000, dur=0.5):
#     """
#     y: [B, T]
#     """
#     ramp = torch.cos(torch.linspace(math.pi, 2 * math.pi, int(fs * dur), device=y.device))
#     ramp = (ramp + 1) / 2
#     y[:,:len(ramp)] *= ramp
#     for i in range(y.shape[0]):
#         y[i,(ilens[i]-len(ramp)):ilens[i]] *= ramp.flip([0])
#     return y

# # def speechweighted_snr(target, noise):
# #     """Apply speech weighting filter to signals and get SNR."""
# #     target_filt = scipy.signal.convolve(
# #         target, SPEECH_FILTER, mode="full", method="fft"
# #     )
# #     noise_filt = scipy.signal.convolve(noise, SPEECH_FILTER, mode="full", method="fft")

# #     # rms of the target after speech weighted filter
# #     targ_rms = np.sqrt(np.mean(target_filt ** 2))

# #     # rms of the noise after speech weighted filter
# #     noise_rms = np.sqrt(np.mean(noise_filt ** 2))

# #     if noise_rms==0:
# #         print('tar', targ_rms, 'noise', noise_rms, 'max amp', np.max(np.abs(noise)), flush=True)
# #         return 0
    
# #     sw_snr = np.divide(targ_rms, noise_rms)
# #     return sw_snr

# def speechweighted_snr(speech, filter):
#     """
#     Apply speech weighting filter to signals and get SNR.
#     input: speech[B, S, M, T], S0=target, filter[1,1,T']
#     output: [B, S-1]
#     """
#     B, S, M, _ = speech.shape
#     speech = torch.nn.functional.pad(speech, (filter.shape[-1] - 1,filter.shape[-1] - 1), 'constant', 0).transpose(1,2).reshape(B*M, S, -1) # [B x M, S, T+2(F-1)]
#     filter = filter.repeat(S, 1, 1) # [S, 1, T']
#     speech_filt = torch.nn.functional.conv1d(speech, filter, groups=S).reshape(B, M, S, -1) # [B, M, S, T]
#     speech_rms = torch.sqrt(torch.mean(speech_filt ** 2, dim=(1, 3)) + EPS) # [B, S]

#     # if noise_rms==0:
#     #     print('tar', targ_rms, 'noise', noise_rms, 'max amp', torch.max(torch.abs(noise)), flush=True)
#     #     return 0
    
#     sw_snr = torch.divide(speech_rms[:,:1], speech_rms[:,1:] + EPS) # [B, S-1]
#     return sw_snr

# def remove_delay_from_rirs(h):
#     device = h.device
#     h = h.cpu().numpy()
#     delay = sys.maxsize

#     for i in range(len(h)):
#         for j in range(len(h[i])):
#             h_env = np.absolute(scipy.signal.hilbert(h[i,j]))
#             # h_env = np.absolute(h[i][j])
#             if delay > np.argmax(h_env):
#                 delay = np.argmax(h_env)

#     h = torch.tensor(h[:,:,delay:], device=device)
#     return h

# def reverberating(spk_wav):
#     """
#     input: [B, S, T]
#     return: [B, M, T]
    
#     [Batch, Cin, T] * [Cout, Cin/group, T] -> [Batch, Cout, T']
#     Batch = 1
#     Cin = B x S
#     Cout = B x S x 2M
#     groups = B x S
#     [1, B x S, T] * [B x S x 2M, (B x S)/(B x S), T] -> [1, B x S x 2M, T']
#     """
#     B, S, T = spk_wav.shape
#     rirs = []
#     max_len = 0
#     for i in range(B):
#         rir, rir_direct = rirgen(device=spk_wav.device) # [M, S, T]
#         rir = torch.cat((rir, rir_direct), dim=0).flip(dims=[-1]) # [2M, S, T]
#         # rir = remove_delay_from_rirs(rir) # [2M, S, newT]
#         # rir = rir.transpose(0, 1).reshape(S*2*rir.shape[0], -1) # [S x 2M, T]
#         rir = rir.transpose(0, 1).reshape(S*rir.shape[0], -1) # [S x 2M, T]
#         rirs.append(rir)
#         max_len = max(max_len, rir.shape[-1])
    
#     M = rirs[0].shape[0] // (S * 2)
#     rirs = [torch.nn.functional.pad(rir, (0, max_len-rir.shape[-1]), 'constant', 0) for rir in rirs]
#     rirs = torch.stack(rirs, dim=0).view(-1, 1, max_len) # [BS x 2M, 1, T_rir]
#     spk_wav = torch.nn.functional.pad(spk_wav, (max_len - 1, 0), 'constant', 0).reshape(1, B*S, -1) # [1, BS, T_wav]
#     try:
#         conv_spk = torch.nn.functional.conv1d(spk_wav, rirs, groups=B*S).view(B, S, 2*M, T) # [B, S, 2M, T]
#     except:
#         rirs = rirs.cpu()
#         spk_wav = spk_wav.cpu()
#         conv_spk = torch.nn.functional.conv1d(spk_wav, rirs, groups=B*S).view(B, S, 2*M, T) # [B, S, 2M, T]
#         conv_spk = conv_spk.cuda()
#     # reverb_spk, anec_spk = torch.sum(conv_spk[:,:,:M], dim=1), conv_spk[:,0,M:]
#     reverb_spk, anec_spk = conv_spk[:,:,:M], conv_spk[:,:,M:] # [B, S, M, T]
#     # print('reverb', torch.sum(torch.abs(reverb_spk), dim=-1), 'anec', torch.sum(torch.abs(anec_spk), dim=-1))
#     return reverb_spk, anec_spk

# # def reverberating(spk_wav):
# #     """
# #     input: [B, T]
# #     return: [B, M, T]
# #     """
# #     B, T = spk_wav.shape
# #     device = spk_wav.device
# #     spk_wav = spk_wav.cpu().numpy()
# #     max_len = 0
# #     reverb_spk = []
# #     for i in range(B):
# #         far = []
# #         # rir, rir_direct = rirgen(device=device) # [M, 1, T]
# #         rir, rir_direct = rirgen() # [M, 1, T]
# #         for chan in range(6):
# #             far.append(scipy.signal.fftconvolve(spk_wav[i], rir[chan,0])[:T])
# #         reverb_spk.append(far)
# #     return torch.from_numpy(np.array(reverb_spk)).cuda(device)

# def mixing(inter_wav, tar_wav, snr_dB,ilens):
#     """
#     inter_wav: [T]
#     snr_dB: float
#     output: [T]
#     """
#     inter_wav = apply_ramp(inter_wav.cpu(),ilens)
#     snr_ref = speechweighted_snr(tar_wav, inter_wav)
#     inter_wav = inter_wav * (snr_ref + EPS)
#     inter_wav = inter_wav * 10 ** ((-snr_dB) / 20)
#     mix_wav = inter_wav + tar_wav
#     all_signals = np.concatenate((mix_wav, tar_wav, inter_wav))
#     norm = np.max(np.abs(all_signals))
#     mix_wav /= norm
#     tar_wav /= norm
#     return mix_wav, tar_wav

# def noreverb_mixing(noise_wav, spk_wav, ilens, filter):
#     """
#     noise_wav: [B, T]
#     spk_wav: [B, S, T]
#     output: [B, T]
#     """
#     B, S, T = spk_wav.shape
#     snr_ref = speechweighted_snr(torch.stack((spk_wav.unsqueeze(2)[:,0], noise_wav.unsqueeze(1)), dim=1), filter) # [B, 1]
#     sir_ref = speechweighted_snr(spk_wav.unsqueeze(2), filter).unsqueeze(-1) # [B, S-1, 1]
#     noise_wav = noise_wav * snr_ref # [B, T]
#     spk_wav[:,1:] = spk_wav[:,1:] * sir_ref # [B, S-1, T]
#     #snr_dB = torch.from_numpy(np.random.rand(B, 1)).float().cuda() * 10 - 5 # [-5, 5]
#     sir_dB = torch.from_numpy(np.random.rand(B, S-1, 1)).float().cuda() * 20 - 15 # [-15, 5]
#     #noise_wav = noise_wav * 10 ** ((-snr_dB) / 20)
#     spk_wav[:,1:] = spk_wav[:,1:] * 10 ** ((-sir_dB) / 20)
    
#     mix_wav = torch.sum(spk_wav, dim=1) + noise_wav # [B, T]
#     return mix_wav, spk_wav[:,0]

# def reverb_mixing(noise_wav, spk_wav, ilens, filter, has_inter):
#     """
#     noise_wav: [B, T]
#     spk_wav: [B, S, T]
#     output: [B, T]
#     """
#     # events = []
#     # for i in range(3):
#     #     events.append({
#     #         'start': torch.cuda.Event(enable_timing=True),
#     #         'end': torch.cuda.Event(enable_timing=True)
#     #     })
#     B, S, T = spk_wav.shape
#     # generate reverb speech
#     # start = time.time()
#     # events[0]['start'].record()
#     reverb_spk, anec_spk = reverberating(spk_wav) # [B, S, M, T]
#     # events[0]['end'].record()
#     # end_reverb = time.time()
    
#     # noise_wav = apply_ramp(noise_wav, ilens)
#     M = reverb_spk.shape[2]
#     noise_wav = noise_wav.view(B, 1, T).repeat(1, M, 1) # [B, M, T]
#     # events[1]['start'].record()
#     snr_ref = speechweighted_snr(torch.stack((reverb_spk[:,0], noise_wav), dim=1), filter).unsqueeze(-1) # [B, 1, 1]
#     # events[1]['end'].record()
#     sir_ref = (has_inter.unsqueeze(1) * speechweighted_snr(reverb_spk, filter)).unsqueeze(-1).unsqueeze(-1) # [B, S-1, 1, 1] # [B, S-1, 1, 1]
#     # events[2]['start'].record()
#     # sir_ref = speechweighted_snr(reverb_spk, filter).unsqueeze(-1).unsqueeze(-1)
#     # events[2]['end'].record()
#     # print('noise', noise_wav.shape, 'snr', snr_ref)
#     noise_wav = noise_wav * snr_ref # [B, M, T]
#     reverb_spk[:,1:] = reverb_spk[:,1:] * sir_ref # [B, S-1, M, T]
#     snr_dB = torch.from_numpy(np.random.rand(B, 1, 1)).float().cuda() * 10 - 5 # [-5, 5]
#     sir_dB = torch.from_numpy(np.random.rand(B, S-1, 1, 1)).float().cuda() * 7 - 2 # [-2, 5]
#     # snr_dB = torch.rand(B, 1, 1, device=noise_wav.device) * 25 - 15 # range [-15 , 10], shape [B, 1, 1], later change to [-10, 10]
#     # sir_dB = torch.rand(B, S-1, 1, 1, device=reverb_spk.device) * 10 - 5 # range [-5, 5], shape [B, S-1, 1, 1], later change to [-15, 5]
#     noise_wav = noise_wav * 10 ** ((-snr_dB) / 20)
#     reverb_spk[:,1:] = reverb_spk[:,1:] * 10 ** ((-sir_dB) / 20)
    
#     mix_wav = torch.sum(reverb_spk, dim=1) + noise_wav # [B, M, T]
#     # mix_wav = torch.sum(reverb_spk, dim=1)
#     # print('reverb', reverb_spk.shape, 'mix', mix_wav.shape, 'noise', noise_wav.shape, 'snr', snr_dB.shape, 'sir', sir_dB.shape)
#     # print('noise', torch.sum(torch.abs(noise_wav), dim=-1), 'spk', torch.sum(torch.abs(reverb_spk), dim=-1))
#     # norm = torch.amax(torch.abs(torch.cat((mix_wav, reverb_spk, noise_wav), dim=1)), dim=(1,2), keepdim=True) # [B, 1, 1]
#     # norm = torch.amax(torch.abs(mix_wav), dim=(1,2), keepdim=True) # [B, 1, 1]
#     # mix_wav = mix_wav / norm * 0.3
#     # reverb_spk = reverb_spk / norm * 0.3
#     # mix_wav = (32767 * mix_wav).astype(np.int16)
#     # spk_wav = (32767 * spk_wav).astype(np.int16)
#     # end_mix = time.time()
#     # return mix_wav, anec_spk[:,0], # anec_spk[:,1], noise_wav, end_reverb - start, end_mix - end_reverb
#     # for i, event in enumerate(events):
#     #     elapsed_time = event['start'].elapsed_time(event['end'])
#     #     print(f"Segment {i+1} elapsed time: {elapsed_time} ms")
#     return mix_wav, reverb_spk[:,0]