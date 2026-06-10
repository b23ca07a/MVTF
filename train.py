import os
import sys
import torch
import torch.nn as nn
import numpy as np
import logging
sys.path.append('../../..')
import argparse
import time
import yaml
import random
from torch.utils.tensorboard import SummaryWriter

from utils.utils import get_instance
from trainer.trainer import Trainer
#from models.sepformer import SBTransformerBlock as SBtf
#from models.sepformer import Dual_Path_Model as DPM
import models.tfgridnet_separator as module_model
import shutil
import dataset
import dataloader

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

import random
# fix random seeds for reproducibility
SEED = 1234
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# torch.cuda.manual_seed_all(SEED)

def main(config, args):
  global IS_MASTER
  IS_MASTER = args.rank == 0
  print("arg.gpu:", args.gpu)
  if args.gpu is not None:
    warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism')
  if args.dist_url == "env://" and args.word_size == -1:
    args.world_size = int(os.environ["WORLD_SIZE"])
  print("world_size is :", args.world_size)
  args.distributed = args.world_size > 1 or args.mutiprocessing_distributed
  print("args.distributed is :", args.distributed)
  ngpus_per_node = torch.cuda.device_count()
  print("ngpus_per_node :", ngpus_per_node)
  print("args.multiprocessing_distributed is :", args.multiprocessing_distributed)
  if args.multiprocessing_distributed:
    args.world_size = ngpus_per_node * args.world_size
    print("world_size:", args.world_size)
    mp.spawn(run, nprocs = ngpus_per_node, args = (ngpus_per_node, args))
  else:
    run(ngpus_per_node, config, args)
  cleanup()


def run(ngpus_per_node, config, args):
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        print("rank is ", args.rank)
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
            print('rank is ',args.rank)
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu

        print("args.dist_url is",args.dist_url)
        print("args.dist-backend is",args.dist_backend)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank) 
        np.random.seed(SEED + args.rank)
        if args.rank == 0:
            writer = SummaryWriter(log_dir=os.path.join('ckpt', args.name, 'tensorboard'))
        else:
            writer = None
    
    trainset = get_instance(dataset, config['trainset'])
    trainloader = get_instance(dataloader, config['trainloader'], trainset, args.world_size)

    testset = get_instance(dataset, config['testset'])
    testloader = get_instance(dataloader, config['testloader'], testset, args.world_size)
    
    data = {'tr_loader': trainloader, 'cv_loader': testloader}
    
    """ intra_model = SBtf(
        num_layers = config['intra_model']['num_layers'],
        d_model = config['intra_model']['d_model'],
        embd_dim = config['intra_model']['embd_dim'],
        nhead = config['intra_model']['nhead'],
        d_ffn = config['intra_model']['d_ffn'],
        dropout = config['intra_model']['dropout'],
        use_positional_encoding = config['intra_model']['use_positional_encoding'],
        norm_before = config['intra_model']['norm_before'],
    ) """
    # encoder = get_instance(module_model, config['encoder'])
    # decoder = get_instance(module_model, config['decoder'])
    # intra_model = get_instance(module_model, config['intra_model'])
    # inter_model = get_instance(module_model, config['inter_model'])
    # model = get_instance(module_model, config['model'], encoder, decoder, intra_model, inter_model)
    
    model = module_model.TFGridNet(**config['tfgridnet_kwargs'])
 
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    ckpt_dir = os.path.join('ckpt', args.name)
    os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copyfile("./models/tfgridnet_separator.py",os.path.join(ckpt_dir,"tfgridnet_separator.py"))
    # make a copy of config in ckpt_dir
    with open(os.path.join(ckpt_dir, 'config.yml'), 'w') as wfile:
        # json.dump(config, wfile, indent=4, sort_keys=False)
        yaml.safe_dump(config, wfile)
        

    model_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = get_instance(torch.optim, config['optimizer'], model_params)
    #lr_scheduler = get_instance(torch.optim.lr_scheduler, config['lr_scheduler'], optimizer)

    # set logger for printing information.
    log_dir = os.path.join('logs', args.name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, time.strftime('%Y-%m-%d-%H%M.log',time.localtime(time.time()))
    )
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger()
    logger.parent = None

    # logger.info(model)
    # logger.info('-' * 50)

    trainer = Trainer(ckpt_dir = ckpt_dir,
                      data = data,
                      model = model,
                      optimizer = optimizer,
                      logger = logger,
                      writer = writer,
                      args = args
    )
    trainer.train()

def cleanup() :
   dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Speech Separation')
    parser.add_argument('--config', default='config/std_train.yml', type=str,
                        help='config file path (default: None)')
    parser.add_argument('--epochs', default=200, type=int, help='Number of maximum epochs')
    parser.add_argument('--half_lr', dest='half_lr', default=1, type=int, help='Halving learning rate when get small improvement')
    parser.add_argument('--early_stop', dest='early_stop', default=1, type=int, help='Early stop training when no improvement for 10 epochs')
    parser.add_argument('--max_norm', default=1, type=float, help='Gradient norm threshold to clip')
    parser.add_argument('--name', default='sep8k', type=str,
                        help='name of the saved model')
    parser.add_argument('--checkpoint', dest='checkpoint', default=1, type=int, help='Enables checkpoint saving of model')
    parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')
    parser.add_argument('--print-freq', default=100, type=int,
                        help='The frequency of printing information.')
    
    parser.add_argument("--world-size", default=-1, type=int,
                      help='number of nodes for distributed training')
    parser.add_argument("--rank", default=-1, type=int,
                        help='ranking within the nodes')
    parser.add_argument("--dist-url", default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training')
    args = parser.parse_args()
    IS_MASTER = False
    # Read config of the whole system.
    assert os.path.isfile(args.config), "No such file: %s" % args.config
    # with open(args.config) as rfile:
    #     config = json.load(rfile)
    with open(args.config) as rfile:
        config = yaml.safe_load(rfile)

    main(config, args)
