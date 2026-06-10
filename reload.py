# import torch
# from collections import OrderedDict
# import json

# cont = torch.load('ckpt/lrs/temp_best.pth.tar', map_location='cpu')
# state_dict = OrderedDict()
# # state_dict = cont['model_state_dict']
# for k,v in cont['model_state_dict'].items():
#     # print(k)
#     name = k.replace('module.', '')
#     print(name)
#     state_dict[name] = v
# cont['epoch'] = 0
# torch.save(cont, '/public/home/qinxy/jz/avse_challenge/attempt/lipselector/chkpt/raw9600/pretrained.pth.tar')

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import logging
sys.path.append('../../..')
import argparse
import time
import json
import random
# from torch.utils.tensorboard import SummaryWriter

from utils.utils import get_instance
from trainer.trainer import Trainer
#from models.sepformer import SBTransformerBlock as SBtf
#from models.sepformer import Dual_Path_Model as DPM
import models.sepformer as module_model

import dataset
import dataloader

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

with open('config/std_train.json') as rfile:
    config = json.load(rfile)

encoder = get_instance(module_model, config['encoder'])
decoder = get_instance(module_model, config['decoder'])
intra_model = get_instance(module_model, config['intra_model'])
inter_model = get_instance(module_model, config['inter_model'])
model = get_instance(module_model, config['model'], encoder, decoder, intra_model, inter_model)
model_params = list(filter(lambda p: p.requires_grad, model.parameters()))
optimizer = get_instance(torch.optim, config['optimizer'], model_params)
# for name, param in model.named_parameters():
#     print(name)
cont = torch.load('ckpt/lrs/temp_best.pth.tar', map_location='cpu')
torch.save({
    'epoch': cont['epoch'],
    'model_state_dict': cont['model_state_dict'],
    'optimizer_state': optimizer.state_dict(),
    'trandom_state': cont['trandom_state'],
    'nrandom_state': cont['nrandom_state']}, 'ckpt/triloss/pretrained.pth.tar')
print('done')