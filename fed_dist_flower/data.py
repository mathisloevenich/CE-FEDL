# Code by Natasha
# Last updated: 2023.12.30

import json
import random
import sys

import numpy as np
import torch
import shutil
import os.path
from torchvision.datasets import MNIST, utils
from fedlab.utils.dataset.partition import CIFAR10Partitioner
from torch.utils.data import DataLoader, Subset, random_split, TensorDataset
from torchvision.datasets import CIFAR10, STL10
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import datasets, transforms


class FEMNIST(MNIST):
	def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
		super(MNIST, self).__init__(root, transform=transform, target_transform=target_transform)
		self.download = download
		self.download_link = 'https://media.githubusercontent.com/media/GwenLegate/femnist-dataset-PyTorch/main/femnist.tar.gz'
		self.file_md5 = 'a8a28afae0e007f1acb87e37919a21db'
		self.train = train
		self.root = root
		self.training_file = f'{self.root}/FEMNIST/processed/femnist_train.pt'
		self.test_file = f'{self.root}/FEMNIST/processed/femnist_test.pt'
		self.user_list = f'{self.root}/FEMNIST/processed/femnist_user_keys.pt'

		if not os.path.exists(f'{self.root}/FEMNIST/processed/femnist_test.pt') \
				or not os.path.exists(f'{self.root}/FEMNIST/processed/femnist_train.pt'):
			if self.download:
				self.dataset_download()
			else:
				raise RuntimeError('Dataset not found, set parameter download=True to download')

		if self.train:
			data_file = self.training_file
		else:
			data_file = self.test_file

		# data_targets_users = torch.load(data_file)
		# self.data, self.targets, self.users = torch.Tensor(data_targets_users[0]), torch.Tensor(data_targets_users[1]), data_targets_users[2]
		# self.user_ids = torch.load(self.user_list)
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
		utils.download_and_extract_archive(
			self.download_link, download_root=f'{self.root}/FEMNIST/raw/', filename=filename, md5=self.file_md5
		)

		files = ['femnist_train.pt', 'femnist_test.pt']
		for file in files:
			# move to processed dir
			shutil.move(os.path.join(f'{self.root}/FEMNIST/raw/', file), f'{self.root}/FEMNIST/processed/')


def femnist_data_json(
		path_to_data_folder="./femnist_data_json",
		num_clients=20,
		subset=50,
		public_ratio=0.5,
		seed=47,
		load_files=10,
		train_bs=32,
		pub_bs=32,
):
	"""
	Input: the path to the folder of json files.

	Data is downloadable from: https://mega.nz/file/XYhhSRIb#PAVgu1zGUoGUU5EzF2xCOnUmGlp5nNQAF8gPdvo_m2U
	It can also be downloaded by cloning the LEAF repository, and running the following command in the femnist folder:
	./preprocess.sh -s niid --iu 1.0 --sf 1.0 -k 0 -t sample --smplseed 42 --spltseed 42

	Returns: a tuple containing the training dataloaders, and test dataloaders,
			with a dataloader for each client
	"""

	all_client_trainloaders = []
	all_client_valloaders = []
	public_x = []
	torch.manual_seed(seed)

	all_clients = []

	for i in tqdm(range(0, load_files)):  # for each json file
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
	grouped_clients = zip(*[iter(all_clients)] * num_clients)

	split_ratio = np.round((1 - public_ratio) / 3, 2)  # metric to help calculate train and val ratios
	train_ratio = 2 * split_ratio  # train data is double the size of the validation data
	val_ratio = 1 - public_ratio - train_ratio  # use remaining data for validation

	print("Pub ratio:", public_ratio, "Val ratio:", val_ratio, "Train ratio:", train_ratio)

	for group in grouped_clients:
		# merge the data arrays together
		x_data = np.concatenate([client[0] for client in group])
		y_data = np.concatenate([client[1] for client in group])

		# split into test and train data
		x_train, x_val, x_pub = random_split(x_data, (train_ratio, val_ratio, public_ratio), torch.Generator().manual_seed(seed))
		y_train, y_test, _ = random_split(y_data, (train_ratio, val_ratio, public_ratio), torch.Generator().manual_seed(seed))

		# put the dataset into dataloaders
		train_loader = DataLoader(dataset=list(zip(x_train, y_train)),
								  batch_size=train_bs,
								  shuffle=True,
								  pin_memory=True)

		val_loader = DataLoader(dataset=list(zip(x_val, y_test)),
								 batch_size=train_bs,
								 shuffle=True,
								 pin_memory=True)

		# add the dataloader to the overall list
		all_client_trainloaders.append(train_loader)
		all_client_valloaders.append(val_loader)
		public_x.append(x_pub)

	# subset the data loaders to the given number
	subset_trainloaders = random.sample(all_client_trainloaders, subset)
	subset_valloaders = random.sample(all_client_valloaders, subset)

	public_tensor = torch.tensor(np.concatenate(public_x, axis=0))

	public_loader = DataLoader(
		dataset=TensorDataset(public_tensor),
		batch_size=pub_bs,
		shuffle=True,
		pin_memory=True
	)

	return subset_trainloaders, subset_valloaders, public_loader


def femnist_data(num_clients=10, public_ratio=0.1, train_bs=32, pub_bs=32):
	mean = (0.9637,)
	std = (0.1591,)

	# mean, std = (0.1307,), (0.3081,)

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
	])

	train_dataset = FEMNIST(
		root="./femnist_data",
		train=True,
		download=True,
		transform=transform
	)
	test_dataset = FEMNIST(
		root="./femnist_data",
		train=False,
		download=True,
		transform=transform
	)
	train_dataset.subsample(0.1)
	test_dataset.subsample(0.1)

	# Split train dataset into train and public datasets
	len_pub = int(public_ratio * len(train_dataset))
	len_train = len(train_dataset) - len_pub
	public_data, train_data = torch.utils.data.random_split(train_dataset, [len_pub, len_train])

	# Split train dataset for each client
	train_samples = len(train_data) // num_clients
	client_train_datasets = [
		Subset(train_data, range(i * train_samples, (i + 1) * train_samples))
		for i in range(num_clients)
	]
	# split test dataset for each client
	val_samples = len(test_dataset) // num_clients
	client_val_datasets = [
		Subset(test_dataset, range(i * val_samples, (i + 1) * val_samples))
		for i in range(num_clients)
	]

	# Create PyTorch DataLoader objects for train and validation datasets of each client
	client_train_loaders = [
		DataLoader(dataset, batch_size=train_bs, shuffle=True) for dataset in client_train_datasets
	]
	client_val_loaders = [
		DataLoader(dataset, batch_size=train_bs, shuffle=False) for dataset in client_val_datasets
	]

	public_loader = torch.utils.data.DataLoader(public_data, batch_size=pub_bs, shuffle=False)

	return client_train_loaders, client_val_loaders, public_loader


def emnist_data(num_clients=10, public_ratio=0.1, train_bs=32, pub_bs=32):
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])

	train_dataset = datasets.EMNIST(
		root="emnist_data",
		split="balanced",
		train=True,
		download=True,
		transform=transform
	)
	test_dataset = datasets.EMNIST(
		root="emnist_data",
		split="balanced",
		train=False,
		download=True,
		transform=transform
	)

	# Split train dataset into train and public datasets
	len_pub = int(public_ratio * len(train_dataset))
	len_train = len(train_dataset) - len_pub
	public_data, train_data = torch.utils.data.random_split(train_dataset, [len_pub, len_train])

	# Split train dataset for each client
	train_samples = len(train_data) // num_clients
	client_train_datasets = [
		Subset(train_data, range(i * train_samples, (i + 1) * train_samples))
		for i in range(num_clients)
	]
	# split test dataset for each client
	val_samples = len(test_dataset) // num_clients
	client_val_datasets = [
		Subset(test_dataset, range(i * val_samples, (i + 1) * val_samples))
		for i in range(num_clients)
	]

	# Create PyTorch DataLoader objects for train and validation datasets of each client
	client_train_loaders = [
		DataLoader(dataset, batch_size=train_bs, shuffle=True) for dataset in client_train_datasets
	]
	client_val_loaders = [
		DataLoader(dataset, batch_size=train_bs, shuffle=False) for dataset in client_val_datasets
	]

	public_loader = torch.utils.data.DataLoader(public_data, batch_size=pub_bs, shuffle=False)

	return client_train_loaders, client_val_loaders, public_loader


def cifar_data(num_clients=10, balanced_data=False, public_ratio=0.1, train_bs=32, pub_bs=32):
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
		balance = True
		partition = "iid"
		dir_alpha = None
	else:  # data not balanced
		balance = None
		partition = "dirichlet"
		dir_alpha = 0.3

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
			batch_size=train_bs,
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
