# Code by Natasha
# Last updated: 2023.12.30

import json
import random

import numpy as np
import torch
from fedlab.utils.dataset.partition import CIFAR10Partitioner
from torch.utils.data import DataLoader, Subset, random_split, TensorDataset
from torchvision.datasets import CIFAR10, STL10
from tqdm import tqdm

import torch
from torchvision import datasets, transforms


def femnist_data(num_clients, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_datasets = []
    test_datasets = []
    public_dataset = datasets.EMNIST(root='./femnist_data', split='byclass', train=True, download=True, transform=transform)

    total_samples = len(public_dataset)
    samples_per_client = total_samples // num_clients

    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client

        client_dataset = torch.utils.data.Subset(public_dataset, range(start_idx, end_idx))
        train_datasets.append(client_dataset)
        test_datasets.append(
            datasets.EMNIST(root='./femnist_data', split='byclass', train=False, download=True, transform=transform))

    train_loaders = [torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in
                     train_datasets]
    test_loaders = [torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False) for dataset in
                    test_datasets]

    return train_loaders, test_loaders, public_dataset


def cifar_data(num_clients=50, balanced_data=False, public_ratio=0.1, train_bs=32, val_bs=32, pub_bs=32):
    """
    Returns: a tuple containing the training data loaders, and test data loaders,
             with a dataloader for each client
    """
    # Download and reshape the dataset
    train_data = CIFAR10(root="cifar_data", train=True, download=True)
    test_data = CIFAR10(root="cifar_data", train=False, download=True)
    x_train = (train_data.data / 255).astype(np.float32).transpose(0, 3, 1, 2)
    y_train = np.array(train_data.targets, dtype=np.int64)
    x_test = (test_data.data / 255).astype(np.float32).transpose(0, 3, 1, 2)
    y_test = np.array(test_data.targets, dtype=np.int64)

    if balanced_data:
        balance=True
        partition="iid"
        dir_alpha=None
    else: # data not balanced
        balance=None
        partition="dirichlet"
        dir_alpha=0.3

    torch.manual_seed(42)

    # Partition the data
    partitioned_train_data = CIFAR10Partitioner(
        train_data.targets, num_clients, balance=balance, partition=partition, dir_alpha=dir_alpha, seed=42
    )
    partitioned_test_data = CIFAR10Partitioner(
        test_data.targets, num_clients, balance=True, partition="iid", seed=42
    )

    train_loaders = []
    val_loaders = []

    public_x = []
    # pubic_y = []

    # Put the data onto a dataloader for each client, following the partitions
    for client in range(num_clients):
        client_x = x_train[partitioned_train_data[client], :, :, :]
        client_y = y_train[partitioned_train_data[client]]

        # now split and put some into public data
        len_pub = int(len(client_x) * public_ratio)
        len_train = len(client_x) - len_pub

        client_x_train = client_x[:len_train]
        client_y_train = client_y[:len_train]

        # append partition to public data
        public_x.append(client_x[len_train:])
        # pubic_y.extend(client_y[len_train:])

        train_loader = DataLoader(dataset=list(zip(client_x_train, client_y_train)),
                                  batch_size=train_bs,
                                  shuffle=True,
                                  pin_memory=True)

        train_loaders.append(train_loader)

        client_x_val = x_test[partitioned_test_data[client], :, :, :]
        client_y_val = y_test[partitioned_test_data[client]]
        val_loader = DataLoader(
            dataset=list(zip(client_x_val, client_y_val)),
            batch_size=val_bs,
            shuffle=True,
            pin_memory=True
        )

        val_loaders.append(val_loader)

    public_tensor = torch.tensor(np.concatenate(public_x, axis=0))

    public_loader = DataLoader(
        dataset=TensorDataset(public_tensor),
        batch_size=pub_bs,
        shuffle=True,
        pin_memory=True
    )

    return train_loaders, val_loaders, public_loader



