import torch
import logging
import os
from torch.distributed import init_process_group

def get_logger(filename, local_rank):
    formatter = logging.Formatter(fmt='[%(asctime)s %(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if filename is not None and local_rank <=0: # only log to file for first GPU
        f_handler = logging.FileHandler(filename, 'a')
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

        stdout_handler = logging.StreamHandler()
        stdout_handler.setFormatter(formatter)
        stdout_handler.setLevel(logging.INFO)
        logger.addHandler(stdout_handler)
    else: # null handlers for other GPUs
        null_handler = logging.NullHandler()
        null_handler.setLevel(logging.INFO)
        logger.addHandler(null_handler)
    
    return logger

def ddp_setup(rank: int, world_size: int, min_rank: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1235" + str(min_rank)
    torch.backends.cudnn.benchmark = True
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def tocuda(rank, data):
    if type(data) is list:
        if len(data) == 1:
            return data[0].to(rank)
        else:
            return [x.to(rank) for x in data]
    else:
        return data.to(rank)
    
def evaluate(rank, model, loader):
        ''' Evaluate some model on some data '''
        ncorrect = 0
        nsamples = 0

        model.eval()
        for *data, target in loader:
            data, target = tocuda(rank, data), tocuda(rank, target)
            with torch.no_grad():
                output = model.module(data)
            pred = output.data.max(1)[1]
            
            ncorrect += pred.eq(target.data).sum().cpu().item()
            nsamples += len(target)

        acc = ncorrect / nsamples
        return acc
    
class DataIterator(object):
    def __init__(self, dataloader):
        assert isinstance(dataloader, torch.utils.data.DataLoader), 'Wrong loader type'
        self.loader = dataloader
        self.iterator = iter(self.loader)

    def __next__(self):
        try:
            x, y = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            x, y = next(self.iterator)

        return x, y
