import numpy as np
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset, random_split
import torch
from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageNet
import os


def build_train_dataloader(args, preprocess=None):
    dataset_name = args.dataset

    if dataset_name == "cifar10":
        train_dataset = CIFAR10(root='./data/dataset', train=True, download=False,
                                transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224))]))
    elif dataset_name == "cifar100":
        if preprocess is not None:
            train_dataset = CIFAR100(root='/mnt/sharedata/ssd3/common/datasets/cifar-100-python', download=False,
                                     train=True, transform=preprocess)
        else:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            train_dataset = CIFAR100(root='/mnt/sharedata/ssd3/common/datasets/cifar-100-python', download=False, train=True, transform=train_transform)


    elif dataset_name == "imagenet":
        # Load datasets
        if preprocess is not None:
            train_dataset = torchvision.datasets.ImageFolder(
                root="/mnt/sharedata/ssd3/common/datasets/imagenet/images/train",
                transform=preprocess
            )
        else:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            train_dataset = torchvision.datasets.ImageFolder(
                root="/mnt/sharedata/ssd3/common/datasets/imagenet/images/train",
                transform=train_transform
            )
    else:
        raise NotImplementedError
    label2class = train_dataset.classes
    if dataset_name == "imagenet":
        label2class = load_label2class(args, "/mnt/sharedata/ssd3/common/datasets/imagenet")
    args.label2class = np.array(label2class)
    args.num_classes = len(label2class)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    return train_loader


def build_cal_test_loader(args, preprocess=None):
    dataset_name = args.dataset

    if dataset_name == "cifar10":
        val_dataset = CIFAR10(root='./data/dataset', train=False, download=True,
                                 transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
        label2class = val_dataset.classes
        val_dataset = Subset(val_dataset, range(0, 100))
    elif dataset_name == "cifar100":
        if preprocess:
            val_dataset = CIFAR100(root='/mnt/sharedata/ssd3/common/datasets/cifar-100-python', download=False,
                                   train=False,
                                   transform=preprocess)
        else:
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            val_dataset = CIFAR100(root='/mnt/sharedata/ssd3/common/datasets/cifar-100-python', download=False, train=False,
                                 transform=val_transform)
        label2class = val_dataset.classes
    elif dataset_name == "imagenet":
        if preprocess:
            val_dataset = torchvision.datasets.ImageFolder(
                root="/mnt/sharedata/ssd3/common/datasets/imagenet/images/val",
                transform=preprocess
            )
        else:
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            val_dataset = torchvision.datasets.ImageFolder(
                root="/mnt/sharedata/ssd3/common/datasets/imagenet/images/val",
                transform=val_transform
            )
        label2class = load_label2class(args,"/mnt/sharedata/ssd3/common/datasets/imagenet")
        print(label2class)
    else:
        raise NotImplementedError

    args.label2class = np.array(label2class)
    args.num_classes = len(label2class)
    if args.algorithm == "standard":
        cal_loader, tune_loader= None, None
        test_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        cal_size = int(len(val_dataset) * args.cal_ratio)
        test_size = len(val_dataset) - cal_size
        cal_dataset, test_dataset = random_split(val_dataset, [cal_size, test_size])


        cal_loader = DataLoader(cal_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, pin_memory=True)

    return cal_loader, test_loader


def load_label2class(args, dataset_path):
    # Path to your classnames.txt (adjust as needed)
    classnames_path = os.path.join(dataset_path, 'classnames.txt')

    label2class = []

    # Case 1: Synset format (n01440764 tench, Tinca tinca)
    if os.path.exists(classnames_path):
        with open(classnames_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Handle different possible formats
                    if line.startswith('n'):  # Synset ID format
                        class_name = line.split(' ', 1)[1].split(',')[0]
                    else:  # Direct class name
                        class_name = line
                    label2class.append(class_name)
    # Case 2: Folder names as classes (common in ImageNet)
    else:
        dataset_root = os.path.join(dataset_path, 'train')
        label2class = sorted([d.name for d in os.scandir(dataset_root) if d.is_dir()])

    return label2class

def split_dataloader(original_dataloader, split_ratio=0.5):
        """
        Splits a DataLoader into two Datasets

        Args:
            original_dataloader (DataLoader): The original DataLoader to split.
            split_ratio (float): The ratio of the first subset (default: 0.5).

        Returns:
            subset1: Training dataset
            subset2: Calibration dataset
        """
        dataset = original_dataloader.dataset
        total_size = len(dataset)

        split_size = int(split_ratio * total_size)

        indices = torch.randperm(total_size)
        indices_subset1 = indices[:split_size]
        indices_subset2 = indices[split_size:]

        subset1 = Subset(dataset, indices_subset1)
        subset2 = Subset(dataset, indices_subset2)

        return subset1, subset2