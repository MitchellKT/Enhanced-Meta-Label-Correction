import torchvision.transforms as transforms
from utils import DataIterator
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from CIFAR.load_corrupted_data_mlg import CIFAR10, CIFAR100

def prepare_data(gold_fraction, corruption_prob, args):
    # Following ideas from Nishi et. al
    transform_silver = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    transform_gold = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    args.num_meta = int(50000 * gold_fraction)

    if args.dataset == 'cifar10':
        num_classes = 10
        
        train_data_gold = CIFAR10(
            root=args.data_path, train=True, meta=True, num_meta=args.num_meta, corruption_prob=corruption_prob,
            corruption_type=args.corruption_type, transform=transform_gold, download=True)
        
        train_data_silver = CIFAR10(
            root=args.data_path, train=True, meta=False, num_meta=args.num_meta, corruption_prob=corruption_prob,
            corruption_type=args.corruption_type, transform=transform_silver, download=True, seed=args.seed)
        
        test_data = CIFAR10(root=args.data_path, train=False, transform=test_transform, download=True)

        valid_data = CIFAR10(
            root=args.data_path, train=True, meta=True, num_meta=args.num_meta, corruption_prob=corruption_prob,
            corruption_type=args.corruption_type, transform=test_transform, download=True)

    elif args.dataset == 'cifar100':
        num_classes = 100
        
        train_data_gold = CIFAR100(
            root=args.data_path, train=True, meta=True, num_meta=args.num_meta, corruption_prob=corruption_prob,
            corruption_type=args.corruption_type, transform=transform_gold, download=True)
        
        train_data_silver = CIFAR100(
            root=args.data_path, train=True, meta=False, num_meta=args.num_meta, corruption_prob=corruption_prob,
            corruption_type=args.corruption_type, transform=transform_silver, download=True, seed=args.seed)
        
        test_data = CIFAR100(root=args.data_path, train=False, transform=test_transform, download=True)

        valid_data = CIFAR100(
            root=args.data_path, train=True, meta=True, num_meta=args.num_meta, corruption_prob=corruption_prob,
            corruption_type=args.corruption_type, transform=test_transform, download=True)    
    
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
    
    valid_loader  = DataLoader(valid_data, batch_size=test_batch_size, shuffle=(val_sampler is None),
                            num_workers=args.prefetch, pin_memory=True, sampler=val_sampler)
    
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=(test_sampler is None),
                            num_workers=args.prefetch, pin_memory=True, sampler=test_sampler)

    return train_gold_loader, train_silver_loader, valid_loader, test_loader, num_classes
