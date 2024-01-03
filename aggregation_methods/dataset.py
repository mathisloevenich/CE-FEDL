import torch
import torchvision
import torchvision.transforms as transforms

from torch import nn, default_generator, randperm
from torchvision.models import resnet18
from torchvision.datasets import MNIST, utils
from torch.utils.data import DataLoader, ConcatDataset, random_split, Dataset
from torch.utils.data.dataset import Subset

import json
import numpy as np
import shutil
import os.path
import PIL
import random

from tqdm import tqdm
from PIL import Image
from abc import abstractmethod


def femnist_data(path_to_data_folder="./data/femnist_data", batch_size=32, combine_clients=20, subset=50, seed=47, load_files=10):
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
    torch.manual_seed(seed)

    if combine_clients <= 1:
        for i in tqdm(range(0, load_files)): # for each json file
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
                    X_train, X_test = random_split(X_data, (0.9, 0.1), torch.Generator().manual_seed(seed))
                    y_train, y_test = random_split(y_data, (0.9, 0.1), torch.Generator().manual_seed(seed))

                    # put the dataset into dataloaders
                    train_loader = DataLoader(dataset=list(zip(X_train, y_train)),
                                              batch_size=batch_size,
                                              shuffle=True,
                                              pin_memory=True)
                    
                    test_loader = DataLoader(dataset=list(zip(X_test, y_test)),
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True)

                    # add the dataloader to the overall list
                    all_client_trainloaders.append(train_loader)
                    all_client_testloaders.append(test_loader)
    else:
        all_clients = []

        for i in tqdm(range(0, load_files)): # for each json file
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
            X_train, X_test = random_split(X_data, (0.9, 0.1), torch.Generator().manual_seed(seed))
            y_train, y_test = random_split(y_data, (0.9, 0.1), torch.Generator().manual_seed(seed))

            # put the dataset into dataloaders
            train_loader = DataLoader(dataset=list(zip(X_train, y_train)),
                                      batch_size=32,
                                      shuffle=True,
                                      pin_memory=True)
            
            test_loader = DataLoader(dataset=list(zip(X_test, y_test)),
                                     batch_size=32,
                                     shuffle=True,
                                     pin_memory=True)

            # add the dataloader to the overall list
            all_client_trainloaders.append(train_loader)
            all_client_testloaders.append(test_loader)
    
    # subset the data loaders to the given number
    subset_trainloaders = random.sample(all_client_trainloaders, subset)
    subset_testloaders = random.sample(all_client_testloaders, subset)
    return subset_trainloaders, subset_testloaders


def femnist_data_2(data_path="./data"):
    """ Work in Process
    
    Input: the path to the folder of json files
    Returns: a tuple containing the training dataloaders, and test dataloaders,
             with a dataloader for each client
    """
    train_data = FEMNIST(root=data_path, train=True, download=True)
    test_data = FEMNIST(root=data_path, train=False, download=True)

    return train_data, test_data

def cifar_data(data_path: str = "./data"):
    """
    Returns: a tuple containing the training data loaders, and test data loaders,
             with a dataloader for each client
    """
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    train_data = torchvision.datasets.CIFAR10(root=data_path, train=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]), download=True)
    test_data = torchvision.datasets.CIFAR10(root=data_path, train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]), download=True)

    return train_data, test_data


def prepare_dataset(num_partitions: int, batch_size: int, dataset_func = cifar_data, val_ratio: float = 0.1):

    train, test = dataset_func()

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


class FEMNIST(MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(MNIST, self).__init__(root, transform=transform, target_transform=target_transform)
        self.download = download
        self.download_link = 'https://media.githubusercontent.com/media/GwenLegate/femnist-dataset-PyTorch/main/femnist.tar.gz'
        self.file_md5 = '60433bc62a9bff266244189ad497e2d7'
        self.train = train
        self.root = root
        self.training_file = f'{self.root}/FEMNIST/processed/femnist_train.pt'
        self.test_file = f'{self.root}/FEMNIST/processed/femnist_test.pt'

        if not os.path.exists(f'{self.root}/FEMNIST/processed/femnist_test.pt') or not os.path.exists(f'{self.root}/FEMNIST/processed/femnist_train.pt'):
            if self.download:
                self.dataset_download()
            else:
                raise RuntimeError('Dataset not found, set parameter download=True to download')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        data_and_targets = torch.load(data_file)
        self.data, self.targets = data_and_targets[0], data_and_targets[1]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='F')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def dataset_download(self):
        paths = [f'{self.root}/FEMNIST/raw/', f'{self.root}/FEMNIST/processed/']
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

        # download files
        filename = self.download_link.split('/')[-1]
        utils.download_and_extract_archive(self.download_link, download_root=f'{self.root}/FEMNIST/raw/', filename=filename, md5=self.file_md5)

        files = ['femnist_train.pt', 'femnist_test.pt']
        for file in files:
            # move to processed dir
            shutil.move(os.path.join(f'{self.root}/FEMNIST/raw/', file), f'{self.root}/FEMNIST/processed/')
