import os
import torch
import numpy as np
import argparse
import yaml
import soundfile as sf
from collections import OrderedDict
from tqdm import tqdm

import models.dcunet as module_model
import librosa
from transform import get_preprocessing_pipelines
from models.tcn.visualfeat_net import VisualFeatNet

def load_pretrained_modules(model, ckpt_path):
    model_info = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in model_info['model_state_dict'].items():
        # print(k)
        name = k.replace("module.", "")    # remove 'module.'
        state_dict[name] = v
    model.load_state_dict(state_dict)

    return model

def _trun_wav(y, tlen, offset=0):
    if y.shape[-1] < tlen:
        npad = tlen - y.shape[-1]
        y = np.pad(y, [(0, 0)]*(y.ndim-1)+[(0, npad)], mode='constant', constant_values=0)
    else:
        y = y[..., offset:offset+tlen]
    return y 

def load_data(mix_wav_path, lip_emb):
    mix_wav, _ = librosa.load(mix_wav_path, sr=16000, mono=False) # [T]
    ilen = mix_wav.shape[-1]
    # lip_emb = np.load(lip_path)["data"]
    # print('lip', lip_emb.shape)
    lip_emb = lip_emb.squeeze(0).cpu().numpy()
    switch = True
    while ilen > len(lip_emb) * 640:
        if switch:
            lip_emb = np.insert(lip_emb, -1, lip_emb[-1,...], axis=0)
            switch = False
        else:
            lip_emb = np.insert(lip_emb, 0, lip_emb[0,...], axis=0)
            switch = True
    mix_wav = _trun_wav(mix_wav, len(lip_emb) * 640)
    assert mix_wav.shape[-1] == len(lip_emb) * 640, 'mix {}, frame len {}'.format(mix_wav.shape, len(lip_emb))
    
    mix_wav = torch.from_numpy(mix_wav) # [M, L] or [L]
    ilen = torch.tensor([ilen])
    lip_emb = torch.from_numpy(lip_emb)
    return mix_wav, ilen, lip_emb

def main(config):
    mix_wav_path = 'infer_one_video/a05.wav'
    lip_path = 'infer_one_video/speaker2.npz'
    dest_path = 'infer_one_video'
    mix_name = mix_wav_path.split('/')[-1].split('.')[0]
    lip_name = lip_path.split('/')[-1].split('.')[0]
    print(mix_name, lip_name)
    os.makedirs(dest_path, exist_ok=True)
    
    lipreading_preprocessing_func = get_preprocessing_pipelines()['test']
    frames = torch.from_numpy(lipreading_preprocessing_func(np.load(lip_path)['data'])).float()
    visnet = VisualFeatNet(pretrain='pretrained/frcnn_128_512.backbone.pth.tar')
    visnet.cuda()
    visnet.eval()
    
    with torch.no_grad():
        frames = frames.cuda()
        # print('frames', frames.shape)
        lip_emb = visnet(frames.unsqueeze(0).unsqueeze(0))
        lip_emb = lip_emb.transpose(-1, -2)
        print(lip_emb.shape)
        np.savez_compressed(os.path.join(dest_path, mix_name+'_'+lip_name+'.npz'), data=lip_emb.cpu())
    print('lip_emb', lip_emb.shape)
    model = module_model.Tar_Model(**config['MaskNet'])
    model = load_pretrained_modules(model, 'ckpt/base/temp_best.pth.tar')
    model.cuda()
    model.eval()

    with torch.no_grad():
        mix_wav, ilens, lip_emb = load_data(mix_wav_path, lip_emb)
        mix_wav = mix_wav.unsqueeze(0).cuda()
        # ilens = ilens.squeeze(-1).cuda()
        lip_emb = lip_emb.unsqueeze(0).cuda()
        print('mix', mix_wav.shape, 'lip', lip_emb.shape)
        est = model(mix_wav, lip_emb) # [B, T]
        print('est', est.shape)
        est = est[..., :ilens[0]] # [B, T]
        sf.write(os.path.join(dest_path, 'dcu_{}_{}.wav'.format(mix_name, lip_name)), est.squeeze().cpu().numpy(), samplerate=16000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Speech Separation')
    parser.add_argument('--config', default='config/std_train_dcunet.yml', type=str,
                        help='config file path (default: None)')
    args = parser.parse_args()
    # Read config of the whole system.
    assert os.path.isfile(args.config), "No such file: %s" % args.config
    with open(args.config) as rfile:
        config = yaml.safe_load(rfile)

    main(config)