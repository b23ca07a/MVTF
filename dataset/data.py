import torch
from torch.utils.data import Dataset
import soundfile
import numpy as np
import random
import pandas as pd
import glob
import librosa
#from emb_extract.inference import sv_model_select
from utils.utils import speechweighted_snr, apply_ramp,mixing
from utils.genrir import RandomRirGenerator
from icecream import ic
class AVSyncDataset(Dataset):
    def __init__(self, meta_file, noise_scp, fs=16000):
        """
        Args:
            meta_file: 每行格式 target_audio inter_audio lip0_path lip1_path
            noise_scp: 噪声文件路径列表
        """
        # 加载元数据
        self.meta_data = [line.strip().split() for line in open(meta_file)]
        self.noise_list = [line.strip().split()[1] for line in open(noise_scp)]
        self.inter_len = len(self.meta_data)
        self.noise_len = len(self.noise_list)
        self.mask_prob = 0.1
        # ic(self.meta_data[0])
        # ic(self.noise_list[0])
        
        # 音频参数
        self.sr = fs
        self.audio_per_frame = 640  # 每唇动帧对应的采样点数
        self.fix_duration = 4       # 固定时长（秒）
        self.total_frames = int(self.fix_duration * self.sr / self.audio_per_frame)  # 100
        self.audio_length = self.total_frames * self.audio_per_frame  # 64000
        # 创建随机数生成器（保证可重复性）
        self.rng = np.random.default_rng(seed=1234)
    def _load_independent_audio(self, path):
      """独立音频加载（用于inter）"""
      full_wav, _ = librosa.load(path, sr=self.sr, mono=True)
      total_samples = len(full_wav)
      # 合并边界条件判断：当音频长度 <= 目标长度时统一处理
      if total_samples <= self.audio_length:
          # 修复点1：去除未定义的start_sample/end_sample，直接使用完整音频
          # 计算需要填充的长度（当长度正好相等时pad_needed=0）
          pad_needed = self.audio_length - total_samples

          # 修复点2：简化填充逻辑，原代码的维度处理适用于多维数组，但音频是单声道一维数组
          padded = np.pad(full_wav, 
                         (0, pad_needed),  # 仅在后端填充
                         mode='constant')

          return torch.from_numpy(padded.astype(np.float32))
      else:
          # 修复点3：确保随机范围有效 (high = total_samples - self.audio_length + 1)
          # 当total_samples == self.audio_length + 1时，允许start=0和start=1两种可能
          max_start = total_samples - self.audio_length
          start = self.rng.integers(0, max_start + 1)  # 包含两端端点

          # 修复点4：简化切片操作，去除不必要的...语法
          return torch.from_numpy(
              full_wav[start:start+self.audio_length].astype(np.float32)
          )
    def _load_core_audio(self, path, start_frame):
        """核心音频加载逻辑（供target/inter复用）"""

        start_sample = start_frame * self.audio_per_frame
        end_sample = start_sample + self.audio_length
        
        # 加载完整音频
        full_wav, _ = librosa.load(path, sr=self.sr, mono=True)
        total_samples = full_wav.shape[-1]
        # 边界处理
        if total_samples < end_sample:
            # 获取有效部分并填充
            valid_part = full_wav[..., max(0, start_sample):]
            pad_needed = end_sample - total_samples
            padded = np.pad(valid_part,
                            [(0,0)]*(full_wav.ndim-1) + [(0, pad_needed)],
                            mode='constant')
            return torch.from_numpy(padded.astype(np.float32))
        else:
            # 正常截取
            return torch.from_numpy(
                full_wav[..., start_sample:end_sample].astype(np.float32)
            )

    def _load_noise(self, path):
        """独立噪声加载（不需要对齐）"""

        noise, _ = librosa.load(path, sr=self.sr, mono=False)
        
        # 随机截取或填充
        if noise.shape[-1] < self.audio_length:
            pad = [(0,0)]*(noise.ndim-1) + [(0, self.audio_length - noise.shape[-1])]
            return torch.from_numpy(np.pad(noise, pad, mode='constant').astype(np.float32))
        else:
            start = self.rng.integers(0, noise.shape[-1] - self.audio_length)
            return torch.from_numpy(noise[..., start:start+self.audio_length].astype(np.float32))

    def _load_lip_features(self, path, start_frame):
        """唇动特征加载（三视角共享逻辑）"""

        lip = np.load(path)["data"].squeeze(0)  # [T, 512]
        T = lip.shape[0]
        
        if T < self.total_frames:
            # 长度不足时从起点填充
            padded = np.pad(lip,
                            [(0, self.total_frames - T), (0,0)],
                            mode='constant')
            return torch.from_numpy(padded.astype(np.float32))
        else:
            # 根据统一起始位置截取
            return torch.from_numpy(
                lip[start_frame:start_frame+self.total_frames].astype(np.float32)
            )

    def __getitem__(self, index):
        # 解析主数据
        index,noise_index = index
        target_path, inter_path, lip0_path, lip1_path,lip2_path = self.meta_data[index]
        #ic(target_path, inter_path, lip0_path, lip1_path)
        # 确定统一起始帧（基于主唇动特征）

        with np.load(lip0_path) as data:
            lip_ref = data["data"].squeeze(0)
        max_start = max(0, lip_ref.shape[0] - self.total_frames)
        start_frame = self.rng.integers(0, max_start) if max_start > 0 else 0
        #print(f"[+] target_path, inter_path, lip0_path, lip1_path,start_frame ={target_path} {inter_path} {lip0_path} {lip1_path} {start_frame}")
        # 同步加载核心数据
        target_audio = self._load_core_audio(target_path, start_frame)
        inter_audio = self._load_independent_audio(inter_path)
        snr_dB = float(self.rng.uniform(-10, 10))
        # print(f"snr_db = {snr_dB}")
        mix,tar = mixing(inter_audio,target_audio,snr_dB)
        lip0 = self._load_lip_features(lip0_path, start_frame)
        lip1 = self._load_lip_features(lip1_path, start_frame)
        lip2 = self._load_lip_features(lip2_path, start_frame)
        #view_mask = (torch.rand(3) < self.mask_prob).float()
        view_mask = torch.zeros(3)
        # view_mask = torch.tensor([0,1,1])
        # view_mask = torch.tensor([1,1,0])

        if view_mask.sum() == 3:
            view_mask[0] = 0
        # view_mask = 1-view_mask
        # 加载噪声（独立随机截取）
        ilen = np.array([self.audio_length])
        ilen = torch.from_numpy(ilen)
        noise_path = self.noise_list[noise_index]
        noise = self._load_noise(noise_path)
        return {
            'mix': mix,
            'tar': tar,
            'noise': noise,
            'lip_0': lip0,
            'lip_1': lip1,
            'lip_2': lip2,
            'view_mask': view_mask,
            'ilens': ilen
        }
    def __len__(self):
        return len(self.meta_data)

        
class inf_dataset(Dataset):
    def __init__(self, mix_scp, lip_scp, fs):
        self.mix = {x.split()[0]:x.split()[1] for x in open(mix_scp)}
        # self.s1 = {x.split()[0]:x.split()[1] for x in open(s1_scp)}
        # self.s2 = {x.split()[0]:x.split()[1] for x in open(s2_scp)}
        self.lip = {x.split()[0]:x.split()[1] for x in open(lip_scp)}
        # assert len(self.mix) == len(self.lip) * 7
        
        #self.sv_model = sv_model_select('ResNetSE34_emb256').sv_model
        mix_id = []
        # s1_id = []
        # s2_id = []
        lip_id = []
        for l in open(mix_scp):
            mix_id.append(l.split()[0])
        # for l in open(s1_scp):
        #     s1_id.append(l.split()[0])
        # for l in open(s2_scp):
        #     s2_id.append(l.split()[0])
        # for l in open(tar_scp):
        #     tar_id.append(l.split()[0])
        # for l in open(lip_scp):
        #     lip_id.append(l.split()[0])
        
        self.mix_id = mix_id
        # self.s1_id = s1_id
        # self.s2_id = s2_id
        # self.tar_id = tar_id
        # self.lip_id = lip_id
        
        self.fs = fs
        self.len = len(self.mix)
        print(self.len)
    
    def _trun_wav(self, y, tlen, offset=0):
        if y.shape[-1] < tlen:
            npad = tlen - y.shape[-1]
            y = np.pad(y, [(0, 0)]*(y.ndim-1)+[(0, npad)], mode='constant', constant_values=0)
        else:
            y = y[..., offset:offset+tlen]
        return y 
    # def _trun_wav(self, y, tlen, offset=0):
    #     if y.shape[0] < tlen:
    #         npad = tlen - y.shape[0]
    #         y = np.pad(y, (0, npad), mode='constant', constant_values=0)
    #     else:
    #         y = y[offset:offset+tlen]
    #     return y 
    
    def __getitem__(self, sample_idx):
        if isinstance(sample_idx, int):
            index, tlen = sample_idx, None
        elif len(sample_idx) == 2:
            index, tlen = sample_idx
        else:
            raise AssertionError
        mix_utt = self.mix_id[index]
        # lip_utt = mix_utt.split('_')[0]+'_'+mix_utt.split('_')[1]
        lip_utt = mix_utt
        # s1_utt = self.s1_id[index]
        # s2_utt = self.s2_id[index]
        
        mix_wav_path = self.mix[mix_utt]
        # s1_wav_path = self.s1[s1_utt]
        # s2_wav_path = self.s2[s2_utt]
        lip_path = self.lip[lip_utt]
        
        mix_wav, _ = librosa.load(mix_wav_path, sr=self.fs, mono=False) # [T]
        # if mix_wav.ndim == 1:
        #     mix_wav = np.repeat(mix_wav.reshape(1,-1), 6, axis=0)
        ilen = mix_wav.shape[-1]
        # print('mix', mix_wav.shape, 'tar', tar_wav.shape)
        # if tar_wav.ndim == 2:
        #     tar_wav = tar_wav[0]
        # s1_wav, _ = librosa.load(s1_wav_path, sr=self.fs, mono=False)
        # if s1_wav.ndim == 2:
        #     s1_wav = s1_wav[0]
        # s2_wav, _ = librosa.load(s2_wav_path, sr=self.fs, mono=False)
        # if s2_wav.ndim == 2:
        #     s2_wav = s2_wav[0]
        #emb_s1 = self.sv_model.embd_extract(aux_wav)
        lip_emb = np.load(lip_path)["data"].squeeze(0)
        switch = True
        while ilen > len(lip_emb) * 640:
            if switch:
                lip_emb = np.insert(lip_emb, -1, lip_emb[-1,...], axis=0)
                switch = False
            else:
                lip_emb = np.insert(lip_emb, 0, lip_emb[0,...], axis=0)
                switch = True
        mix_wav = self._trun_wav(mix_wav, len(lip_emb) * 640)
        
        # if mix_wav.shape[-1] >= tlen:
        #     offset = np.random.randint(0, mix_wav.shape[-1] - tlen)
        #     lip_head = max(int(offset / 640), 0)
        #     lip_tail = lip_head + int(tlen / 640)
        #     try:
        #         lip_emb = lip_emb[lip_head:lip_tail]
        #     except:
        #         print('len', len(tar_wav), 'offset', offset, 'vr', N, 'head', lip_head, 'tail', lip_tail)
        #     mix_wav = self._trun_wav(mix_wav, tlen, offset)
        #     tar_wav = self._trun_wav(tar_wav, tlen, offset)
        #     source_len = np.array([tlen])
        # else:
        #     lip_emb = np.concatenate((lip_emb, np.zeros((int(tlen/640)-N, E)).astype(lip_emb.dtype)), axis=0)
        #     source_len = np.array([mix_wav.shape[-1]])
        #     mix_wav = self._trun_wav(mix_wav, tlen)
        #     tar_wav = self._trun_wav(tar_wav, tlen)
        #     # print('109', mix_wav.shape)
        # assert tar_wav.shape[-1] // 640 == len(lip_emb) and mix_wav.shape[-1] == tar_wav.shape[-1] and mix_wav.shape[-1] == tlen, '{} mix {}, frame len {}'.format(mix_utt, mix_wav.shape, len(lip_emb))
        assert mix_wav.shape[-1] == len(lip_emb) * 640, '{} mix {}, frame len {}'.format(mix_utt, mix_wav.shape, len(lip_emb))
        
        mix_wav = torch.from_numpy(mix_wav) # [M, L] or [L]
        ilen = np.array([ilen])
        ilen = torch.from_numpy(ilen)
        # source = [s1_wav, s2_wav]
        # source = [tar_wav]
        # source = np.asarray(source)
        # source = torch.from_numpy(source) # [2, L]
        # source = torch.from_numpy(tar_wav) # [M, L]
        # source_len = np.array([source.shape[-1]])
        # source_len = torch.from_numpy(source_len)
        #print(source.shape)
        #print(source_len)
        #s1_wav = torch.from_numpy(s1_wav)
        #s2_wav = torch.from_numpy(s2_wav)
        #aux_wav = torch.from_numpy(aux_numpy)
        #emb_s1 = torch.from_numpy(emb_s1)
        lip_emb = torch.from_numpy(lip_emb)
        return mix_utt, mix_wav, ilen, lip_emb,ilen
    
    def __len__(self):
        return self.len