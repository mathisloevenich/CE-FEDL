import torch
import torchvision
from torch import nn
from torchvision.models import resnet18
from torch.utils.data import DataLoader, random_split
from fedlab.utils.dataset.partition import CIFAR10Partitioner
import json
import numpy as np
from tqdm import tqdm

###### Model and dataset utility functions 
###### Example use:

###### import utils
###### train_loaders, test_loaders = utils.femnist_data()

def femnist_model():
    """
    Returns: the Resnet-18 model for the FEMNIST data
    """
    torch.manual_seed(47) # set manual seed so the model always has the same initial parameters
    mnist_model = resnet18(num_classes=62) # 62 classes for femnist -> 26 lower case letters + 26 upper case letters + 10 numbers
    mnist_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # change number of channels to 1
    return mnist_model

def cifar_model():
    """
    Returns: the Resnet-18 model for the CIFAR data
    """
    torch.manual_seed(47) # set manual seed so the model always has the same initial parameters
    cifar_model = resnet18(num_classes=10) # 10 cifar classes
    return cifar_model

def femnist_data(path_to_data_folder="femnist_data"):
    """
    Input: the path to the folder of json files 
    Returns: a tuple containing the training dataloaders, and test dataloaders,
             with a dataloader for each client
    """

    all_client_trainloaders = []
    all_client_testloaders = []

    for i in tqdm(range(0, 36)): # for each json file
        with open(f"{path_to_data_folder}/all_data_{i}.json") as file:

            # load the 100 clients in each json file
            data = json.load(file)
            all_clients = data["users"]

            for client in all_clients:
                # load the dataset from one client
                X_data = data["user_data"][client]["x"]
                num_samples = len(X_data)
                X_data = np.array(X_data).reshape(num_samples, 1, 28, 28) # reshape into BxCxHxW
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
                
    return all_client_trainloaders, all_client_testloaders

def cifar_data():
    """
    Returns: a tuple containing the training data loaders, and test data loaders,
             with a dataloader for each client
    """
    # Download and reshape the dataset
    train_data = torchvision.datasets.CIFAR10(root="cifar_data", train=True, download=True)
    test_data = torchvision.datasets.CIFAR10(root="cifar_data", train=False, download=True)
    X_train = (train_data.data / 255).astype(np.float32).transpose(0, 3, 1, 2)
    y_train = np.array(train_data.targets, dtype=np.int64)
    X_test = (test_data.data / 255).astype(np.float32).transpose(0, 3, 1, 2)
    y_test = np.array(test_data.targets, dtype=np.int64)
    
    # Partition the data into an imbalanced and non-iid form
    partitioned_train_data = CIFAR10Partitioner(train_data.targets,
                                              750,
                                              balance=None,
                                              partition="dirichlet",
                                              dir_alpha=0.3,
                                              seed=42)
    partitioned_test_data = CIFAR10Partitioner(test_data.targets,
                                           750,
                                           balance=True,
                                           partition="iid",
                                           seed=42)
    
    all_client_trainloaders = []
    all_client_testloaders = []

    # Put the data onto a dataloader for each client, following the partitions
    for client in range(750):
        client_X_train = X_train[partitioned_train_data[client], :, :, :]
        client_y_train = y_train[partitioned_train_data[client]]
        torch.manual_seed(47)
        train_loader = DataLoader(dataset=list(zip(client_X_train, client_y_train)),
                                  batch_size=32,
                                  shuffle=True,
                                  pin_memory=True)
        client_X_test = X_test[partitioned_test_data[client], :, :, :]
        client_y_test = y_test[partitioned_test_data[client]]
        torch.manual_seed(47)
        test_loader = DataLoader(dataset=list(zip(client_X_test, client_y_test)),
                                  batch_size=32,
                                  shuffle=True,
                                  pin_memory=True)

        all_client_trainloaders.append(train_loader)
        all_client_testloaders.append(test_loader)
        
    return all_client_trainloaders, all_client_testloaders