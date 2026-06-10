# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:00:46 2020

@author: yoonsanghyu
"""

# Created on 2018/12
# Author: Kaituo XU

from itertools import permutations
import torch
import yaml

from .losses import stoi_loss, pesq_loss
from models.stft_encoder import STFTEncoder

with open('config/std_train.yml') as rfile:
    config = yaml.safe_load(rfile)

STFT_ENC = STFTEncoder(**config['stft_kwargs'])
EPS = 1e-8
def sisnr(x, s, eps):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + EPS)
    return 20 * torch.log10(EPS + l2norm(t) / (l2norm(x_zm - t) + EPS))


def batchMean_sisnrLoss(est, clean):
    batch_sisnr = sisnr(est, clean, EPS)
    return -torch.mean(batch_sisnr)

# def sdr(est, ref, eps=EPS):

#     if est.shape != ref.shape:
#         raise RuntimeError(f"Shape mismatch: {est.shape} vs {ref.shape}")
#     ref_pow = torch.sum(ref ** 2, dim=-1)
#     distortion_pow = torch.sum((est - ref) ** 2, dim=-1)
#     sdr_val = 10 * torch.log10(ref_pow / (distortion_pow + eps) + eps)
#     return sdr_val
# def sdr(est, ref, eps=1e-8):
#     ref_pow = torch.sum(ref ** 2, dim=-1)
#     distortion_pow = torch.sum((est - ref) ** 2, dim=-1)
#     sdr_val = 10 * torch.log10(ref_pow / (distortion_pow + eps) + eps)
#     return sdr_val
def sdr(est, ref, eps=1e-8):
    # 可选：减去直流分量
    # est = est - torch.mean(est, dim=-1, keepdim=True)
    # ref = ref - torch.mean(ref, dim=-1, keepdim=True)
    
    ref_pow = torch.sum(ref ** 2, dim=-1)
    distortion_pow = torch.sum((est - ref) ** 2, dim=-1)
    sdr_val = 10 * torch.log10(ref_pow / (distortion_pow + eps) + eps)
    return sdr_val
def batch_mean_sdr_loss(est, clean):
    batch_sdr = sdr(est, clean)
    return -torch.mean(batch_sdr)
def batchMean_WavMagLoss(est, clean, ilens):
    batch_wav_mag = wav_mag_mc(est, clean, ilens, EPS)
    return -torch.mean(batch_wav_mag)

def batchMean_sisnr_se_mcLoss(est, clean):
    batch_sisnr = sisnr_se_mc(est, clean, EPS)
    return -torch.mean(batch_sisnr)

def sisnr_se_mc(x, s, eps):
    """
    x: [B, T]
    s: [B, T]
    ilens: [B]
    """
    x = x.squeeze(1)
    s = s.squeeze(1)
    N = x.shape[-1]
    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)
    
    def l1norm(mat, keepdim=False):
        return torch.norm(mat, p=1, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True) # [B, T]
    s_zm = s - torch.mean(s, dim=-1, keepdim=True) # [B, T]
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * x_zm / (l2norm(x_zm, keepdim=True)**2 + eps)
    sisnr_se = 20 * torch.log10(eps + l2norm(s_zm) / (l2norm(s_zm - t) + eps))
    wav_mc = torch.mean(torch.abs(x_zm - s_zm), dim=-1) # [B]
    
    return sisnr_se - wav_mc

def wav_mag_mc(x, s, ilens, eps):
    """
    x: [B, T]
    s: [B, T]
    ilens: [B]
    """
    x = x.squeeze(1)
    s = s.squeeze(1)
    N = x.shape[-1]
    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)
    
    def l1norm(mat, keepdim=False):
        return torch.norm(mat, p=1, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True) # [B, T]
    s_zm = s - torch.mean(s, dim=-1, keepdim=True) # [B, T]
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * x_zm / (l2norm(x_zm, keepdim=True)**2 + EPS)
    sisnr_se = 20 * torch.log10(EPS + l2norm(s_zm) / (l2norm(s_zm - t) + EPS))
    wav_mc = torch.norm(t, p=1, dim=-1, keepdim=False) / N
    
    x_zm_complex = STFT_ENC(x_zm.unsqueeze(-1), ilens)[0] # [B, T, 1, F]
    s_zm_complex = STFT_ENC(s_zm.unsqueeze(-1), ilens)[0] # [B, T, 1, F]
    x_zm_mag = x_zm_complex.abs().squeeze(2) # [B, T, F]
    s_zm_mag = s_zm_complex.abs().squeeze(2) # [B, T, F]
    mag_diff = torch.abs(x_zm_mag) - torch.abs(s_zm_mag)
    mag_mc = torch.norm(mag_diff, p=1, dim=(-2,-1), keepdim=False) / (ilens * N)
    # print(sisnr_se, wav_mc, mag_mc)
    
    return sisnr_se - mag_mc - wav_mc
    

def cal_loss(source, estimate_source, source_lengths):
    """
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B]
    """
    max_snr, perms, max_snr_idx = cal_wav_mag_mc_with_pit(source,
                                                      estimate_source,
                                                      source_lengths)
    loss = 0 - torch.mean(max_snr)
    reorder_estimate_source = reorder_source(estimate_source, perms, max_snr_idx)
    # loss = loss + 0.5*stoi_loss(reorder_estimate_source.flatten(end_dim=1), source.flatten(end_dim=1), device=source.device) \
    #             + 0.5*pesq_loss(reorder_estimate_source.flatten(end_dim=1), source.flatten(end_dim=1), device=source.device)
    
    return loss, max_snr, estimate_source, reorder_estimate_source

def cal_wav_mag_mc_with_pit(source, estimate_source, source_lengths):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    # mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    # mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
    # zero_mean_target = source - mean_target
    # zero_mean_estimate = estimate_source - mean_estimate
    est_std_ = torch.std(estimate_source, dim=(1, 2), keepdim=True)  # [B, 1, 1]
    zero_mean_estimate = estimate_source / est_std_  # RMS normalization [B, C, T]
    zero_mean_target = source / est_std_ # [B, C, T]
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask
    mix_wav_diff = torch.sum(zero_mean_estimate - zero_mean_target, dim=1) # [B, T]
    
    zm_tar_complex, flens = STFT_ENC(zero_mean_target.transpose(1, 2), source_lengths) # [B, T, C, F], [B]
    zm_est_complex, _ = STFT_ENC(zero_mean_estimate.transpose(1, 2), source_lengths) # [B, T, C, F], [B]
    zm_tar_mag = zm_tar_complex.abs().transpose(1, 2) # [B, C, T, F]
    zm_est_mag = zm_est_complex.abs().transpose(1, 2) # [B, C, T, F]
    mix_mag_diff = torch.abs(torch.sum(zm_est_mag, dim=1)) - torch.abs(torch.sum(zm_tar_mag, dim=1)) # [B, T, F]
    
    # Step 2. Wav+Mag+MC with PIT
    s_target = torch.unsqueeze(zero_mean_target, dim=1) # [B, 1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2) # [B, C, 1, T]
    s_wav_diff = s_estimate - s_target # [B, C, C, T]
    pair_wise_wav = torch.sum(torch.abs(s_wav_diff), dim=3) / num_samples # [B, C, C]
    
    s_target_mag = torch.unsqueeze(zm_tar_mag, dim=1) # [B, 1, C, T, F]
    s_estimate_mag = torch.unsqueeze(zm_est_mag, dim=2) # [B, C, 1, T, F]
    s_mag_diff = s_estimate_mag - s_target_mag # [B, C, C, T, F]
    pair_wise_mag = torch.sum(torch.abs(s_mag_diff), dim=(3, 4)) / s_mag_diff.shape[4] / flens.unsqueeze(1).unsqueeze(2) # [B, C, C]
    
    pair_wise_both = pair_wise_wav + pair_wise_mag

    # Get max_wav_mag with mixture constraint of each utterance
    # permutations, [C!, C]
    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # one-hot, [C!, C, C]
    index = torch.unsqueeze(perms, 2)
    perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    # [B, C!] <- [B, C, C] einsum [C!, C, C], wav_mag sum of each permutation
    wav_mag_set = torch.einsum('bij,pij->bp', [pair_wise_both, perms_one_hot]) # [B, C!]
    # print('pair', pair_wise_both.shape, 'one hot', perms_one_hot)
    mc_wav = torch.sum(torch.abs(mix_wav_diff), dim=1) / source_lengths # [B]
    mc_mag = torch.sum(torch.abs(mix_mag_diff), dim=(1, 2)) / (source_lengths * mix_mag_diff.shape[-1]) # [B]
    wav_mag_set = wav_mag_set - mc_wav.unsqueeze(1) - mc_mag.unsqueeze(1) # [B, C!]
    
    max_wav_mag_idx = torch.argmax(wav_mag_set, dim=1)  # [B]
    print('perms', perms.shape, 'idx', max_wav_mag_idx) # [C!, C], [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_wav_mag, _ = torch.max(wav_mag_set, dim=1, keepdim=True)
    max_wav_mag /= C
    return max_wav_mag, perms, max_wav_mag_idx

def cal_sisnr_se_mc_with_pit(source, estimate_source, source_lengths):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask
    wav_diff = torch.sum(zero_mean_estimate - zero_mean_target, dim=1) # [B, T]

    # Step 2. SI-SNR-SE with PIT
    # reshape to use broadcast
    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, C, C, 1]
    s_estimate_energy = torch.sum(s_estimate ** 2, dim=3, keepdim=True) + EPS  # [B, C, 1, 1]
    pair_wise_proj = pair_wise_dot * s_estimate / s_estimate_energy  # [B, C, C, T]
    # e_noise = s' - s_target
    e_noise = s_target - pair_wise_proj  # [B, C, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(s_target ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]
    # pair_wise_noise = torch.sum(e_noise, dim=3) / num_samples # [B, C, C]

    # Get max_snr with mixture constraint of each utterance
    # permutations, [C!, C]
    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # one-hot, [C!, C, C]
    index = torch.unsqueeze(perms, 2)
    perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
    snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
    mc = torch.sum(torch.abs(wav_diff), dim=1) / source_lengths
    snr_mc_set = snr_set - mc
    max_snr_mc_idx = torch.argmax(snr_mc_set, dim=1)  # [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr_mc, _ = torch.max(snr_mc_set, dim=1, keepdim=True)
    max_snr_mc /= C
    return max_snr_mc, perms, max_snr_mc_idx

def cal_si_snr_with_pit(source, estimate_source, source_lengths):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, C, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]

    # Get max_snr of each utterance
    # permutations, [C!, C]
    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # one-hot, [C!, C, C]
    index = torch.unsqueeze(perms, 2)
    perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
    snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
    max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    max_snr /= C
    return max_snr, perms, max_snr_idx


def reorder_source(source, perms, max_snr_idx):
    """
    Args:
        source: [B, C, T]
        perms: [C!, C], permutations
        max_snr_idx: [B], each item is between [0, C!)
    Returns:
        reorder_source: [B, C, T]
    """
    B, C, *_ = source.size()
    # [B, C], permutation whose SI-SNR is max of each utterance
    # for each utterance, reorder estimate source according this permutation
    max_snr_perm = torch.index_select(perms, dim=0, index=max_snr_idx)
    # maybe use torch.gather()/index_select()/scatter() to impl this?
    reorder_source = torch.zeros_like(source)
    for b in range(B):
        for c in range(C):
            reorder_source[b, c] = source[b, max_snr_perm[b][c]]
    return reorder_source


def get_mask(source, source_lengths):
    """
    Args:
        source: [B, C, T]
        source_lengths: [B]
    Returns:
        mask: [B, 1, T]
    """
    B, _, T = source.size()
    mask = source.new_ones((B, 1, T))
    for i in range(B):
        mask[i, :, source_lengths[i]:] = 0
    return mask


if __name__ == "__main__":
    '''
    torch.manual_seed(123)
    B, C, T = 2, 3, 12
    # fake data
    source = torch.randint(4, (B, C, T))
    estimate_source = torch.randint(4, (B, C, T))
    source[1, :, -3:] = 0
    estimate_source[1, :, -3:] = 0
    source_lengths = torch.LongTensor([T, T-3])
    print('source', source)
    print('estimate_source', estimate_source)
    print('source_lengths', source_lengths)
    
    loss, max_snr, estimate_source, reorder_estimate_source = cal_loss(source, estimate_source, source_lengths)
    print('loss', loss)
    print('max_snr', max_snr)
    print('reorder_estimate_source', reorder_estimate_source)
    '''
