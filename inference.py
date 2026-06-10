import os
os.environ['CUDA_VISIBLE_DEVICES'] ='1'
import torch
import numpy as np
import argparse
import yaml
import soundfile as sf
from collections import OrderedDict
from tqdm import tqdm
from utils.utils import SPEECH_FILTER, mixing
from utils.utils import get_instance
import models.tfgridnet_separator as module_model
from utils.loss.pit_criterion import batchMean_sisnrLoss,batchMean_sisnr_se_mcLoss

from utils.loss.pit_criterion import *
from utils.utils import *
from utils.evaluation import *
import dataset
import dataloader
# import pesq

filter = SPEECH_FILTER.cuda()
def load_pretrained_modules(model, ckpt_path):
    model_info = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in model_info['model_state_dict'].items():
        # print(k)
        name = k.replace("module.", "")    # remove 'module.'
        state_dict[name] = v
    model.load_state_dict(state_dict)

    return model
from icecream import ic
def run(config,epoch):
    inferset = get_instance(dataset, config['inferset'])
    inferloader = get_instance(dataloader, config['inferloader'], inferset)
    print(f"infer = {config['inferset']['args']['meta_file']}")
    model = module_model.TFGridNet(**config['tfgridnet_kwargs'])
    model = load_pretrained_modules(model, './ckpt/25_NEW_random_repeat_conv_before_LSTM_front_product_two_masked_3_random_view_mead_add_after_vconv_new_exceptM019_9/epoch'+epoch+'.pth.tar')
    model.cuda()
    model.eval()
    total_loss_pesq = torch.tensor(0.).cuda()
    total_loss_sdr = torch.tensor(0.).cuda()
    total_loss_stoi = torch.tensor(0.).cuda()
    dest_path = './dest_infer'
    os.makedirs(dest_path, exist_ok=True)
    os.makedirs(dest_path+"/test_mix/", exist_ok=True)
    os.makedirs(dest_path+"/target/", exist_ok=True)
    os.makedirs(dest_path+"/est/", exist_ok=True)
    print("???")
    # factor = [-0.15,-0.16,-0.17,-0.18,-0.19,-0.20]
    # for f in factor:
    #     print(f"factor = {f}")
    with torch.no_grad():
        prog_bar = tqdm(enumerate(inferloader))
        for i, (data) in enumerate(prog_bar):  
            data = data[1]
            mix = data['mix'].cuda()
            tar = data['tar'].cuda()
            B,_ = tar.shape
            # ic(tar.shape)
            # ic(mix.shape)
            ilens = data['ilens'].squeeze(-1).cuda() # [B]
            lip_emb_0 = data['lip_0'].cuda()
            lip_emb_1 = data['lip_1'].cuda()
            # lip_emb_1 = torch.zeros(lip_emb_1.shape).cuda()
            lip_emb_2 = data['lip_2'].cuda()
            # lip_emb_2 = torch.zeros(lip_emb_2.shape).cuda()
            masked = data['view_mask'].cuda()
            padded_noise = data['noise'].cuda()

            est = -model(mix.transpose(1, -1), ilens, lip_emb_0,lip_emb_1,lip_emb_2,masked)[0]*0.2# [B, C, L]
            sf.write(os.path.join(dest_path+"/est/", '{}.wav'.format(str(i))), est.squeeze().cpu().numpy(), samplerate=16000)
            sf.write(os.path.join(dest_path+"/test_mix/", '{}_mix.wav'.format(str(i))), mix.squeeze().cpu().numpy(), samplerate=16000)
            sf.write(os.path.join(dest_path+"/target/", '{}_tar.wav'.format(str(i))), tar.squeeze().cpu().numpy(), samplerate=16000)

    return total_loss_sdr/(i+1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Speech Separation')
    parser.add_argument('--config', default='config/std_train.yml', type=str,
                        help='config file path (default: None)')
    args = parser.parse_args()

    assert os.path.isfile(args.config), "No such file: %s" % args.config
    with open(args.config) as rfile:
        config = yaml.safe_load(rfile)
    # views = ['fff','ttt','ddd','3l30','3r30','3r60','3l60']
    views=['fff']
    # views=['3l30']
    # views = ['fl30r30','fl30r60','fl60r60','fr30r30','fr60r60']
    # views.reverse()
    # views = ['fl30r30','fl30r60','fl60r60']#'fr30r30','fr60r60']
    # views = ['fr30r30','fr60r60']
    # for view in views:
    #     config['inferset']['args']['meta_file'] = "/public/home/qinxy/photon/dataset/MEAD_SCP/"+view+"_view_test.scp"
    #     main(config)
    _max = 0
    for i in range(77,78):
        # cnt = 0
        for view in views:
            config['inferset']['args']['meta_file'] = "/public/home/qinxy/photon/dataset/MEAD_SCP/"+view+"_view_test.scp"
            # cnt+=run(config,str(i))
            run(config,str(i))
        # if(cnt>_max):
        #     _max=cnt
        #     print(f"epoch ======================================{i}")
        #     print(f"avg_value = {cnt/5}")
    # run(config,"77")