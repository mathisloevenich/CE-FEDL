# Code by Natasha
# Last updated: 2023.12.30

import json
import random

import numpy as np
import torch
from fedlab.utils.dataset.partition import CIFAR10Partitioner
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from tqdm import tqdm


def femnist_data(path_to_data_folder="femnist_data", combine_clients=20, subset=50):
    """
    Input: the path to the folder of json files.
    
    Data is downloadable from: https://mega.nz/file/XYhhSRIb#PAVgu1zGUoGUU5EzF2xCOnUmGlp5nNQAF8gPdvo_m2U
    It can also be downloaded by cloning the LEAF repository, and running the following command in the femnist folder:
    ./preprocess.sh -s niid --iu 1.0 --sf 1.0 -k 0 -t sample --smplseed 42 --spltseed 42
    
    Returns: a tuple containing the training dataloaders, and test dataloaders,
             with a dataloader for each client
    """

    all_client_trainloaders = []
    all_client_testloaders = []

    if combine_clients <= 1:
        for i in tqdm(range(0, 36)): # for each json file
            with open(f"{path_to_data_folder}/all_data_{i}.json") as file:

                # load the 100 clients in each json file
                data = json.load(file)
                all_clients = data["users"]

                for client in all_clients:
                    # load the dataset from one client
                    X_data = data["user_data"][client]["x"]
                    num_samples = len(X_data)
                    X_data = np.array(X_data, dtype=np.float32).reshape(num_samples, 1, 28, 28) # reshape into BxCxHxW
                    y_data = np.array(data["user_data"][client]["y"], dtype=np.int64)

                    # split into test and train data
                    X_train, X_test = random_split(X_data, (0.9, 0.1), torch.Generator().manual_seed(42))
                    y_train, y_test = random_split(y_data, (0.9, 0.1), torch.Generator().manual_seed(42))

                    # put the dataset into dataloaders
                    torch.manual_seed(47)
                    train_loader = DataLoader(dataset=list(zip(X_train, y_train)),
                                              batch_size=32,
                                              shuffle=True,
                                              pin_memory=True)
                    torch.manual_seed(47)
                    test_loader = DataLoader(dataset=list(zip(X_test, y_test)),
                                             batch_size=32,
                                             shuffle=True,
                                             pin_memory=True)

                    # add the dataloader to the overall list
                    all_client_trainloaders.append(train_loader)
                    all_client_testloaders.append(test_loader)
    else:
        all_clients = []

        for i in tqdm(range(0, 36)): # for each json file
            with open(f"{path_to_data_folder}/all_data_{i}.json") as file:
                
                # load the 100 clients in each json file
                data = json.load(file)
                for client in data["users"]:
                    X_data = data["user_data"][client]["x"]
                    num_samples = len(X_data)
                    X_data = np.array(X_data, dtype=np.float32).reshape(num_samples, 1, 28, 28)
                    y_data = np.array(data["user_data"][client]["y"], dtype=np.int64)

                    all_clients.append((X_data, y_data))
                    
        # group the given number of clients together
        grouped_clients = zip(*[iter(all_clients)] * combine_clients)
        for group in grouped_clients:

            # merge the data arrays together
            X_data = np.concatenate([client[0] for client in group])
            y_data = np.concatenate([client[1] for client in group])

            # split into test and train data
            X_train, X_test = random_split(X_data, (0.9, 0.1), torch.Generator().manual_seed(42))
            y_train, y_test = random_split(y_data, (0.9, 0.1), torch.Generator().manual_seed(42))

            # put the dataset into dataloaders
            torch.manual_seed(47)
            train_loader = DataLoader(dataset=list(zip(X_train, y_train)),
                                      batch_size=32,
                                      shuffle=True,
                                      pin_memory=True)
            torch.manual_seed(47)
            test_loader = DataLoader(dataset=list(zip(X_test, y_test)),
                                     batch_size=32,
                                     shuffle=True,
                                     pin_memory=True)

            # add the dataloader to the overall list
            all_client_trainloaders.append(train_loader)
            all_client_testloaders.append(test_loader)
    
    # subset the data loaders to the given number
    random.seed(47)
    subset_trainloaders = random.sample(all_client_trainloaders, subset)
    random.seed(47)
    subset_testloaders = random.sample(all_client_testloaders, subset)
    return subset_trainloaders, subset_testloaders


def cifar_data(num_clients=50, balanced_data=False, public_data_ratio=0.2):
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

    torch.manual_seed(47)

    # Partition the data
    partitioned_train_data = CIFAR10Partitioner(train_data.targets,
                                                  num_clients,
                                                  balance=balance,
                                                  partition=partition,
                                                  dir_alpha=dir_alpha,
                                                  seed=42)

    all_client_trainloaders = []

    # Put the data onto a dataloader for each client, following the partitions
    for client in range(num_clients):
        client_x_train = x_train[partitioned_train_data[client], :, :, :]
        client_y_train = y_train[partitioned_train_data[client]]
        train_loader = DataLoader(dataset=list(zip(client_x_train, client_y_train)),
                                  batch_size=32,
                                  shuffle=True,
                                  pin_memory=True)

        all_client_trainloaders.append(train_loader)

    client_dataset_size = int(len(x_train) * public_data_ratio)

    eval_loader = DataLoader(
        dataset=list(zip(x_test[:client_dataset_size], y_test[:client_dataset_size])),
        batch_size=32, shuffle=True, pin_memory=True
    )

    public_loader = DataLoader(
        dataset=list(zip(x_train[:client_dataset_size], y_train[:client_dataset_size])),
        batch_size=32, shuffle=True, pin_memory=True
    )

    return all_client_trainloaders, eval_loader, public_loader
