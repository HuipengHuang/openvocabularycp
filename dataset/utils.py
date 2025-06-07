import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset, random_split
import torch
from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageNet


def build_dataset(args):
    dataset_name = args.dataset

    if dataset_name == "cifar10":
        train_dataset = CIFAR10(root='./data/dataset', train=True, download=True, transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor()]))
        cal_test_dataset = CIFAR10(root='./data/dataset', train=True, download=False,
                                 transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor()]))
        label2class = train_dataset.classes
        cal_test_dataset = Subset(cal_test_dataset, range(0, 200))
    elif dataset_name == "cifar100":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = CIFAR100(root="./data/dataset", download=True, train=True, transform=train_transform)
        cal_test_dataset = CIFAR100(root='./data/dataset', download=True, train=False,
                                 transform=val_transform)
        label2class = train_dataset.classes
    elif dataset_name == "imagenet":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load datasets
        train_dataset = torchvision.datasets.ImageFolder(
            root="/data/dataset/imagenet/images/train",
            transform=train_transform
        )

        cal_test_dataset = torchvision.datasets.ImageFolder(
            root="/data/dataset/imagenet/images/val",
            transform=val_transform
        )
        #label2class = ImageNet.__class__
    if args.algorithm != "standard":
        cal_size = int(len(cal_test_dataset) * args.cal_ratio)
        test_size = len(cal_test_dataset) - cal_size
        cal_dataset, test_dataset = random_split(cal_test_dataset, [cal_size, test_size])
    else:
        cal_dataset, test_dataset = None, cal_test_dataset
    args.label2class = label2class
    return train_dataset, cal_dataset, test_dataset, label2class


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