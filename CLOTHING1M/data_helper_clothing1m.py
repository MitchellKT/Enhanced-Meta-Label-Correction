import torchvision
import torchvision.transforms as transforms
from utils import DataIterator
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

def _fix_cls_to_idx(ds):
    for cls in ds.class_to_idx:
        ds.class_to_idx[cls] = int(cls)

def prepare_data(args):
    num_classes = 14

    # resnet recommended normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # transform
    # Note: rescaling to 224 and center-cropping already processed in img folders    
    transform_val = transforms.Compose([
        transforms.ToTensor(), # to [0,1]
        normalize
    ])

    transform_strong = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(),
        transforms.ToTensor(), # to [0,1]
        normalize
    ])
    
    train_data_gold = torchvision.datasets.ImageFolder(args.data_path + 'clothing1m/clean_train', transform=transform_strong)
    train_data_silver = torchvision.datasets.ImageFolder(args.data_path + 'clothing1m/noisy_train', transform=transform_val)
    val_data = torchvision.datasets.ImageFolder(args.data_path + 'clothing1m/clean_val', transform=transform_val)
    test_data = torchvision.datasets.ImageFolder(args.data_path + 'clothing1m/clean_test', transform=transform_val)

    # fix class idx to equal to class name
    _fix_cls_to_idx(train_data_gold)
    _fix_cls_to_idx(train_data_silver)
    _fix_cls_to_idx(val_data)
    _fix_cls_to_idx(test_data)

    gold_sampler = DistributedSampler(dataset=train_data_gold, shuffle=True)
    silver_sampler = DistributedSampler(dataset=train_data_silver, shuffle=True)
    val_sampler = None
    test_sampler = None
    test_batch_size = args.test_bs
    eff_batch_size = args.bs // args.n_gpus

    train_gold_loader = DataIterator(DataLoader(train_data_gold, batch_size=eff_batch_size, drop_last=True,
                                                num_workers=args.prefetch, pin_memory=True, sampler=gold_sampler))
    
    train_silver_loader = DataLoader(train_data_silver, batch_size=eff_batch_size, drop_last=True,
                                    num_workers=args.prefetch, pin_memory=True, sampler=silver_sampler)
    
    val_loader  = DataLoader(val_data, batch_size=test_batch_size, shuffle=(val_sampler is None),
                            num_workers=args.prefetch, pin_memory=True, sampler=val_sampler)
    
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=(test_sampler is None),
                            num_workers=args.prefetch, pin_memory=True, sampler=test_sampler)

    return train_gold_loader, train_silver_loader, val_loader, test_loader, num_classes
