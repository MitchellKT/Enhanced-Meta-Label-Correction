import pickle
import argparse
import os

import torch.multiprocessing as mp
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import get_logger, ddp_setup
from trainer import Trainer
from models import *       
from meta_models import *
from CLOTHING1M.data_helper_clothing1m import prepare_data

parser = argparse.ArgumentParser(description='EMLC Training Framework')

# General and paths
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--data_seed', type=int, default=0)
parser.add_argument('--runid', default='clothing1m_run_best', type=str)
parser.add_argument('--data_path', default='data/', type=str, help='Root for the datasets.')
parser.add_argument('--logdir', type=str, default='runs', help='Log folder.')

# Training
parser.add_argument('--epochs', '-e', type=int, default=3, help='Number of epochs to train.')
parser.add_argument('--every', default=10, type=int, help='Eval interval')
parser.add_argument('--bs', default=1024, type=int, help='batch size')
parser.add_argument('--test_bs', default=100, type=int, help='batch size')
parser.add_argument('--gold_bs', type=int, default=1024)
parser.add_argument('--embedding_dim', type=int, default=2048, help='Feature extractor output dim')
parser.add_argument('--label_embedding_dim', type=int, default=128, help='Label embedding dim')
parser.add_argument('--mlp_hidden_dim', type=int, default=128, help='MLP hidden layer units')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optimizer')
parser.add_argument('--main_lr', default=1e-1, type=float, help='lr for main net')
parser.add_argument('--meta_lr', default=1e-1, type=float, help='lr for meta net')
parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--gradient_steps', default=1, type=int, help='Number of look-ahead gradient steps for meta-gradient')
parser.add_argument('--sched_milestones', default='1,5', type=str, help='Milestones in which LR is decreased')
parser.add_argument('--sched_gamma', default=0.1, type=float, help='Multiply LR by gamma upon reaching a scheduled milestone')

# Hardware
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--n_gpus', default=8, type=int)

args = parser.parse_args()

# //////////////// set logging and model outputs /////////////////
def set_logging(rank):
    filename = '_'.join(['clothing1m', args.runid, str(args.epochs), str(args.seed), str(args.data_seed)])
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    logfile = 'logs/' + filename + '.log'
    logger = get_logger(logfile, rank)

    if not os.path.isdir('models'):
        os.mkdir('models')

    logger.info(args)
    return logger

# //////////////////////// defining model ////////////////////////
def build_models(rank, num_classes):
    main_net = generalized_resnet50_clothing(num_classes, args)
    meta_backbone = generalized_resnet50_clothing(num_classes, args)
        
    meta_net = ResNetFeatures(meta_backbone)
    enhancer = TeacherEnhancer(num_classes, args.embedding_dim, args.label_embedding_dim, args.mlp_hidden_dim)

    main_net = main_net.to(rank)
    main_net = DDP(main_net, device_ids=[rank])

    meta_net = meta_net.to(rank)
    meta_net = DDP(meta_net, device_ids=[rank])

    enhancer = enhancer.to(rank)
    enhancer = DDP(enhancer, device_ids=[rank])
    
    return main_net, meta_net, enhancer

# //////////////////////// run experiments ////////////////////////
def run(rank):
    ddp_setup(rank, world_size=args.n_gpus, min_rank=args.gpuid)
    logger = set_logging(rank)
    filename = '_'.join(['clothing1m', args.runid, str(args.epochs), str(args.seed), str(args.data_seed)])
    exp_id = filename

    results = {}

    gold_loader, silver_loader, valid_loader, test_loader, num_classes = prepare_data(args)
    main_net, meta_net, enhacner = build_models(rank, num_classes)

    trainer = Trainer(rank, args, main_net, meta_net, enhacner, gold_loader, silver_loader, valid_loader, test_loader, num_classes, logger, exp_id)
    trainer.train()
    test_acc = trainer.final_eval()
    
    if rank == 0:
        results['method'] = test_acc
        logger.info(' '.join(
            ['Method acc:', str(results['method'])]))
        logger.info('')

    if rank == 0:
        with open('out/' + filename, 'wb') as file:
            pickle.dump(results, file)
    logger.info("Dumped results_ours in file: " + filename)
    destroy_process_group()


if __name__ == "__main__":
    gpus = range(args.gpuid, args.gpuid+args.n_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in gpus)
    mp.spawn(run, nprocs=args.n_gpus)