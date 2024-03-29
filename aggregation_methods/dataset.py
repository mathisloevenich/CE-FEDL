import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data

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


def femnist_data_json(path_to_data_folder="./data/femnist_data", batch_size=32, combine_clients=20, subset=10, seed=47, load_files=10):
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
                    X_data = torch.tensor(np.array(X_data, dtype=np.float32).reshape(num_samples, 1, 28, 28)) # reshape into BxCxHxW
                    y_data = torch.tensor(np.array(data["user_data"][client]["y"], dtype=np.int64))

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
                    X_data = torch.tensor(np.array(X_data, dtype=np.float32).reshape(num_samples, 1, 28, 28))
                    y_data = y_data = torch.tensor(np.array(data["user_data"][client]["y"], dtype=np.int64))

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


def femnist_data(data_path="./data", seed=47):
    """ 
    Input: the path to the folder of json files
    Returns: a tuple containing the training dataloaders, and test dataloaders,
             with a dataloader for each client
    """
    mean = (0.9637,)
    std = (0.1591,)
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])
    train_data = FEMNIST(root=data_path, train=True, download=True, transform=transform)
    test_data = FEMNIST(root=data_path, train=False, download=True, transform=transform)
    train_data.subsample(0.1)
    test_data.subsample(0.1)

    return train_data, test_data


def cifar_data(data_path: str = "./data"):
    """
    Returns: a tuple containing the training data loaders, and test data loaders,
             with a dataloader for each client
    """
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])
    train_data = torchvision.datasets.CIFAR10(root=data_path, train=True, transform=transform, download=True)
    test_data = torchvision.datasets.CIFAR10(root=data_path, train=False, transform=transform, download=True)

    return train_data, test_data


def prepare_dataset(num_partitions: int, batch_size: int, dataset_func = cifar_data, val_ratio: float = 0.1, seed=47):

    train, test = dataset_func()

    num_images = len(train) // num_partitions
    partition_len = [num_images] * num_partitions
    remainder = len(train) % num_partitions
    if remainder > 0:
        partition_len[-1] += remainder

    train = random_split(train, partition_len, torch.Generator().manual_seed(seed))

    train_loaders = []
    val_loaders = []

    for data in train:
        num_total = len(data)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(data, [num_train, num_val], torch.Generator().manual_seed(seed))

        train_loaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=6))
        val_loaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=6))

    test_loader = DataLoader(test, batch_size=batch_size, num_workers=6)

    return train_loaders, val_loaders, test_loader


def femnist_dirichlet(alpha, clients, batch_size, data_path="./data", seed=47):
    torch.manual_seed(seed)
    mean = (0.9637,)
    std = (0.1591,)
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])
    train_data = FEMNIST(root=data_path, train=True, download=True)
    test_data = FEMNIST(root=data_path, train=False, download=True, transform=transform)
    train_data.subsample(0.1)
    test_data.subsample(0.1)

    ytrain_label = np.array([train_data[i][1] for i in range(len(train_data))])

    train_loaders = []
    valid_loaders = []
    net_dataidx_map,traindata_cls_counts = partition(alpha, clients , ytrain_label)
    
    splits=[[] for i in range(clients)]
    for mindex,classes in enumerate(splits):
            for i in traindata_cls_counts[mindex]:
                if traindata_cls_counts[mindex][i]>1:classes.append(i) #if images number <=1, then ignore the parameters of this class in the last layer when ensemble.
    
    #data partition
    for i in range(clients):
        traindata = MNIST_truncated(train_data,net_dataidx_map[i],transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]),)
        
        train, valid = random_split(traindata, (0.9, 0.1))
        valid_loaders.append(torch.utils.data.DataLoader(dataset=valid, batch_size=batch_size, shuffle=True))
        train_loaders.append(torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True))

    test_loaders = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return train_loaders, valid_loaders, test_loaders


def partition(alphaa, n_netss, y_train,):
    min_size = 0
    n_nets = n_netss
    N = y_train.shape[0]
    net_dataidx_map = {}
    alpha = alphaa
    K=10
    while min_size < 10:
        idx_batch = [[] for _ in range(n_nets)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            ## Balance
            proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return net_dataidx_map,traindata_cls_counts


def record_net_data_stats(y_train, net_dataidx_map):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    return net_cls_counts


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
    
    def subsample(self, fraction):
        """ Reduce the size of the dataset to the specified fraction """
        total_samples = len(self.data)
        reduced_size = int(total_samples * fraction)
        indices = random.sample(range(total_samples), reduced_size)
        self.data = self.data[indices]
        self.targets = self.targets[indices]

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


class MNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        data = self.root.data
        target = self.root.targets

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
     
        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, target)

    def __len__(self):
        return len(self.data)
