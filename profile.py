from thop import profile

import os
import torch
import numpy as np
import argparse
import yaml
import soundfile as sf
from collections import OrderedDict
from tqdm import tqdm

from utils.utils import get_instance
# import models.dcunet as module_model
# import models.dcunetV2 as module_modelV2
import models.tfgridnet_separator as module_model

import dataset
import dataloader

def load_pretrained_modules(model, ckpt_path):
    model_info = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in model_info['model_state_dict'].items():
        # print(k)
        name = k.replace("module.", "")    # remove 'module.'
        state_dict[name] = v
    model.load_state_dict(state_dict)

    return model

def main(config):

    model = module_model.TFGridNet(**config['tfgridnet_kwargs'])
    # model = load_pretrained_modules(model, 'ckpt/noattn/temp_best.pth.tar')
    model.cuda()
    model.eval()

    with torch.no_grad():
        mixture = torch.rand(1, 16000).cuda()
        ilens = torch.tensor([16000]).cuda()
        lip_emb = torch.rand(1, 25, 512).cuda()
        
        flops, params = profile(model, (mixture, ilens, lip_emb), verbose=False)
        print(f'FLOPS = {flops / 1000 ** 3} G')
        print(f'Params = {params / 1000 ** 2} M')

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