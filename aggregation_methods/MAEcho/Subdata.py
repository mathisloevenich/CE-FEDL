import numpy as np
import torch 
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import os
import random
import shutil


from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, CIFAR10, utils

torch.manual_seed(0)
torch.cuda.manual_seed(0) 
np.random.seed(0) 


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


def femnist_data(data_path=r'C:\Users\tim\Documents\Guided Project\Pytorch\CE-FEDL\aggregation_methods\data', seed=47):
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


def prepare_dataset(num_partitions: int, batch_size: int, dataset_func = cifar_data, seed=47):

    train, test = dataset_func()

    num_images = len(train) // num_partitions
    partition_len = [num_images] * num_partitions
    remainder = len(train) % num_partitions
    if remainder > 0:
        partition_len[-1] += remainder

    train = random_split(train, partition_len, torch.Generator().manual_seed(seed))

    train_loaders = []

    for data in train:
        train_loaders.append(DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=6))

    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=6)

    return train_loaders, test_loader

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
    
class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None,transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        data = self.root.data
        target =np.array( self.root.targets)
#         target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


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

        
# class SubDataset(Dataset):

#     def __init__(self, original_dataset, sub_labels,aera=[0,1], target_transform=None):
#         super().__init__()
#         self.dataset = original_dataset
#         self.sub_indeces = []
        
#         for index in range(len(self.dataset)):
#             if index<len(self.dataset)*aera[0]  or index>len(self.dataset)*aera[1]:
#                 continue
#             if hasattr(original_dataset, "train_labels"):
                
#                 if self.dataset.target_transform is None:
#                     label = self.dataset.train_labels[index]
#                 else:
#                     label = self.dataset.target_transform(self.dataset.train_labels[index])
#             elif hasattr(self.dataset, "test_labels"):
#                 if self.dataset.target_transform is None:
#                     label = self.dataset.test_labels[index]
#                 else:
#                     label = self.dataset.target_transform(self.dataset.test_labels[index])
#             else:
#                 label = self.dataset[index][1]
#             if label in sub_labels:
#                 self.sub_indeces.append(index)
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.sub_indeces)

#     def __getitem__(self, index):
#         sample = self.dataset[self.sub_indeces[index]]
#         if self.target_transform:
#             target = self.target_transform(sample[1])
#             sample = (sample[0], target)
#         return sample
    
