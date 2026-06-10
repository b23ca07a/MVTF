import torch
import torch.nn.functional as F

nr_of_hz_bands_per_bark_band_16k = [ # 线性谱转bark谱
    1,    1,    1,    1,    1,   1,    1,    1,    2,    1,
    1,    1,    1,    1,    2,   1,    1,    2,    2,    2,
    2,    2,    2,    2,    2,   3,    3,    3,    3,    4,
    3,    4,    5,    4,    5,   6,    6,    7,    8,    9,
    9,    12,   12,   15,   16,  18,   21,   25,   21
]

pow_dens_correction_factor_16k = [ # 由线性谱获取bark谱时，bark谱每个子带的权重
    100.000000,     99.999992,     100.000000,    100.000008,
    100.000008,     100.000015,    99.999992,     99.999969,
    50.000027,      100.000000,    99.999969,     100.000015,
    99.999947,      100.000061,    53.047077,     110.000046,
    117.991989,     65.000000,     68.760147,     69.999931,
    71.428818,      75.000038,     76.843384,     80.968781,
    88.646126,      63.864388,     68.155350,     72.547775,
    75.584831,      58.379192,     80.950836,     64.135651,
    54.384785,      73.821884,     64.437073,     59.176456,
    65.521278,      61.399822,     58.144047,     57.004543,
    64.126297,      54.311001,     61.114979,     55.077751,
    56.849335,      55.628868,     53.137054,     54.985844,
    79.546974
]

abs_thresh_power_16k = [ # loudness mapping
    51286152.00,  2454709.500,  70794.593750,
    4897.788574,  1174.897705,  389.045166,
    104.712860,   45.708820,    17.782795,
    9.772372,     4.897789,     3.090296,
    1.905461,     1.258925,     0.977237,
    0.724436,     0.562341,     0.457088,
    0.389045,     0.331131,     0.295121,
    0.269153,     0.257040,     0.251189,
    0.251189,     0.251189,     0.251189,
    0.263027,     0.288403,     0.309030,
    0.338844,     0.371535,     0.398107,
    0.436516,     0.467735,     0.489779,
    0.501187,     0.501187,     0.512861,
    0.524807,     0.524807,     0.524807,
    0.512861,     0.478630,     0.426580,
    0.371535,     0.363078,     0.416869,
    0.537032
]

width_of_band_bark_16k = [ # predefined weighting for bark spectrum bins
    0.157344,     0.317994,     0.322441,     0.326934,     0.331474,
    0.336061,     0.340697,     0.345381,     0.350114,     0.354897,
    0.359729,     0.364611,     0.369544,     0.374529,     0.379565,
    0.384653,     0.389794,     0.394989,     0.400236,     0.405538,
    0.410894,     0.416306,     0.421773,     0.427297,     0.432877,
    0.438514,     0.444209,     0.449962,     0.455774,     0.461645,
    0.467577,     0.473569,     0.479621,     0.485736,     0.491912,
    0.498151,     0.504454,     0.510819,     0.517250,     0.523745,
    0.530308,     0.536934,     0.543629,     0.550390,     0.557220,
    0.564119,     0.571085,     0.578125,     0.585232
]
# comp_pesq('/home/lhf/tmp/voicebank_demand/test_clean_small/p232_002.wav', '/home/lhf/tmp/voicebank_demand/test_noisy_small/p232_002.wav')

Sp_16k=6.910853e-006  # for bark calc
Sl_16k=1.866055e-001  # for loudness mapping
# Sp_16k=4e4
# Sl_16k=7e-7


class PESQLOSS(object):

  def __init__(self, device):
    self.device = device
    self.bark_mat = self.get_bark_mat().to(device)

  def get_bark_mat(self):
    end_arr = []
    tmp_arr = []

    for i in range(len(nr_of_hz_bands_per_bark_band_16k)):
        w = [0 for j in range(257)]
        if i == 0:
            end_arr.append(nr_of_hz_bands_per_bark_band_16k[i])
        else:
            end_arr.append(end_arr[i-1]+nr_of_hz_bands_per_bark_band_16k[i])
        j_end = end_arr[i]
        j_start = j_end - nr_of_hz_bands_per_bark_band_16k[i] # [j_start, j_end) is bark band

        for j in range(257):
            if j>=j_start and j<j_end:
                w[j]=1
        tmp_arr.append(w)

    t = torch.tensor(tmp_arr, dtype=torch.float32).T.unsqueeze(0)
    return t

  def bark_spec_F(self, spec, bins=49):
    '''
    spec: [B, T, 257]
    '''
    device = spec.device
    assert bins==49, "bark bins is not 49, which is not supported now."
    # print(spec.shape, bark_mat.shape)
    # print(spec.device)
    bark_spec = torch.matmul(spec, self.bark_mat) #[B, T, 49]

    b_w2 = torch.tensor([[pow_dens_correction_factor_16k]]).to(device)
    bark_spec = bark_spec * b_w2 * Sp_16k

    # print("bark_spec", bark_spec)
    # print(bark_spec.mean(1))
    return bark_spec # [ B, T, 49]

  def __call__(self, enc_mag, clean_mag):
    '''
    enc_mag: [B, T, F]
    clean_mag: [B, T, F]
    '''
    # fixed power level
    # target_power_avg = 1e7
    # enc_global_scale = torch.sqrt(target_power_avg/enc_mag.sum())
    # clean_global_scale = torch.sqrt(target_power_avg/clean_mag.sum())
    # enc_mag = enc_mag * enc_global_scale
    # clean_mag = clean_mag * clean_global_scale
    # print("global scale", enc_global_scale, clean_global_scale)

    device = self.device

    # bark trans
    bark_bins = 49
    bn = self.bark_spec_F(enc_mag, bark_bins)
    bc = self.bark_spec_F(clean_mag, bark_bins)

    # frequency power equal
    pn = torch.mean(bn, dim=1, keepdim=True) # [B, 1, Fb]
    pc = torch.mean(bc, dim=1, keepdim=True) # [B, 1, Fb]
    c1 = 1e3 # 1e3
    # print("pn, pc" ,pn, pc)
    ec = (pn + c1)/(pc + c1) * bc # [B, T, Fb]

    # frame power equal
    gn = torch.sum(bn, dim=2, keepdim=True) # [B, T, 1]
    gc = torch.sum(ec, dim=2, keepdim=True)
    c2 = 5e3 # 5e3
    s = (gc + c2)/(gn + c2) * 0.8 # [B, T, 1]
    s[:, 1:, :] += 0.25 * s[:, :-1, :] # 正确做法是迭代
    # print(s)
    en = s * bn

    # loudness mapping
    ec = ec / 10
    en = en / 10
    threshold = torch.tensor(abs_thresh_power_16k).unsqueeze(0).unsqueeze(0).to(device)
    # print(ec, en)
    lc = (threshold / 0.5) * ((0.5 + 0.5 * ec / threshold) - 1) * Sl_16k
    ln = (threshold / 0.5) * ((0.5 + 0.5 * en / threshold) - 1) * Sl_16k
    # print("lc, ln", lc, ln)

    # disturbance  processing
    lcn = torch.cat([lc.unsqueeze(dim=0), ln.unsqueeze(dim=0)], dim=0) # [2, B, T, Fb]
    dz = 0.25 * torch.min(lcn, dim=0, keepdim=False).values # [B, T, Fb]
    # print(dz)
    dis = torch.clamp_min(lc-ln-dz, 0) + torch.clamp_max(lc-ln+dz, 0)
    # print("dis", dis)

    w_bark = torch.tensor(width_of_band_bark_16k).to(device)
    fd = (1./w_bark.sum(dim=-1) * torch.sum((w_bark*dis)**2, dim=-1))**(0.5)  # [B, T]
    fd = torch.sum(w_bark, dim=-1) * fd
    # print("fd", fd)

    hm = ((bn+50)/(bc+50))**1.2
    hm = torch.clamp(hm, 1, 18)
    # print(hm.max())
    afd = (1./w_bark.sum(dim=-1) * torch.sum((w_bark*dis*hm)**2, dim=-1))**0.5  # [B, T]
    afd = torch.sum(w_bark, dim=-1) * afd

    # aggregation
    B, T = fd.shape
    seg_fd = F.unfold(fd.view(B,1,1,T),(1,20),stride=10) # [B, S, 20]
    psqm_sym = torch.mean(seg_fd**6, dim=-1)**(1/6) # [B, S]
    dsym = (torch.mean(psqm_sym**2, dim=-1)**1.5).mean() # scaler

    seg_afd = F.unfold(afd.view(B,1,1,T),(1,20),stride=10) # [B, S, 20]
    psqm_asym = torch.mean(seg_afd**6, dim=-1)**(1/6) # [B, S]
    dasym = (torch.mean(psqm_asym**2, dim=-1)**0.5).mean() # scaler

    # print(dsym, dasym)

    pesq_s = 4.5-0.1*dsym-0.0309*dasym

    return pesq_s


if __name__ == "__main__":
  pass