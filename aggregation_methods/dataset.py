import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize
from torchvision.datasets import MNIST


def get_mnist(data_path: str = './data'):

    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train = MNIST(data_path, train=True, download=True, transform=tr)
    test = MNIST(data_path, train=False, download=True, transform=tr)

    return train, test


def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):

    train, test = get_mnist()

    num_images = len(train) // num_partitions
    partition_len = [num_images] * num_partitions
    train = random_split(train, partition_len, torch.Generator().manual_seed(2023))

    train_loaders = []
    val_loaders = []
    for data in train:
        num_total = len(data)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(data, [num_train, num_val], torch.Generator().manual_seed(2023))

        train_loaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        val_loaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))

    test_loader = DataLoader(test, batch_size=128)

    return train_loaders, val_loaders, test_loader
