import torch 
from torch.utils.data import Dataset
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.datasets import MNIST, CIFAR10
torch.manual_seed(0)
torch.cuda.manual_seed(0) 
np.random.seed(0) 

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
    
from torchvision.datasets import MNIST, utils
from PIL import Image
import os.path
import torch


class FEMNIST(MNIST):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """
    resources = [
        ('https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz',
         '59c65cec646fc57fe92d27d83afdf0ed')]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets, self.users_index = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='F')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def download(self):
        """Download the FEMNIST data if it doesn't exist in processed_folder already."""
        import shutil

        if self._check_exists():
            return

        utils.makedir_exist_ok(self.raw_folder)
        utils.makedir_exist_ok(self.processed_folder)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            utils.download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')
        shutil.move(os.path.join(self.raw_folder, self.training_file), self.processed_folder)
        shutil.move(os.path.join(self.raw_folder, self.test_file), self.processed_folder)

        
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
    
