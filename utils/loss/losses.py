import torch
# from .pesq_loss import PESQLOSS
from .pmsqe import SingleSrcPMSQE
from .stoi_loss import NegSTOILoss


def sisnr(x, s, eps=1e-8):
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
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))


def batchMean_sisnrLoss(est, clean, eps=1e-8):
    batch_sisnr = sisnr(est, clean, eps)
    return -torch.mean(batch_sisnr)


def batchSum_MSE(y1, y2, _idx=2):
    # y1, y2: [N, C, F, T] or [N, F, T]
    loss = (y1-y2) ** _idx
    loss = torch.mean(torch.sum(loss, -2))
    return loss


def batchSum_relativeMSE(y1, y2, RL_epsilon=0.1, index_=2.0):
    # y1, y2: [N, C, F, T] ot [N, F, T]
    relative_loss = torch.abs(y1-y2) / (torch.abs(y1) + torch.abs(y2) + RL_epsilon)
    loss = torch.pow(relative_loss, index_)
    loss = torch.mean(torch.sum(loss, -2))
    return loss


def pesq_loss(ests, refs, device='cuda'):
    enc_mag, clean_mag = mag(ests, refs)
    # required mag shape: [B, T, F]
    # pesqloss_fn = PESQLOSS(device=device)
    pesqloss_fn = SingleSrcPMSQE().to(device)
    batch_pesq = pesqloss_fn(enc_mag, clean_mag)
    return -torch.mean(batch_pesq)


def stoi_loss(est_wav_batch, tar_wav_batch, sr=16000,
              stoi_vad=True, stoi_extend=False,
              stoi_resample=True, device='cuda'):
    '''
    est_wav_batch: [batch, L]
    tar_wav_batch: [batch, L]
    return:        [batch, ]
    '''
    stoiloss_fn = NegSTOILoss(sample_rate=sr, use_vad=stoi_vad,
                            extended=stoi_extend, do_resample=stoi_resample).to(device)
    return torch.mean(stoiloss_fn(est_wav_batch, tar_wav_batch))

def mag(ests, refs, device="cuda", frame_len=256, frame_hop=128, eps=1e-8):
        '''return [B, T, F]'''
        ests_mag = torch.stft(
            input=ests,
            n_fft=frame_len,
            hop_length=frame_hop,
            window=torch.hann_window(frame_len).to(device),
            center=False,
            return_complex=False
        )
        refs_mag = torch.stft(
            input=refs,
            n_fft=frame_len,
            hop_length=frame_hop,
            window=torch.hann_window(frame_len).to(device),
            center=False,
            return_complex=False
        )
        ests_energy = torch.sum(ests_mag**2, -1).squeeze(-1).transpose(1, 2)
        refs_energy = torch.sum(refs_mag**2, -1).squeeze(-1).transpose(1, 2)
        ests_mag = torch.sqrt(ests_energy + eps)
        refs_mag = torch.sqrt(refs_energy + eps)
        return ests_mag, refs_mag