from torch.utils.data import DataLoader

from models import create_model
from utils import train

import torch
import random


class DistillationClient:

    def __init__(
            self,
            cid,
            train_loader: DataLoader,
            x_pub,
            model_architecture="cnn500k",
            dataset_name="cifar"):
        self.cid = cid
        self.train_loader = train_loader
        self.x_pub = x_pub  # fixed public data
        self.soft_labels = None
        self.model = None
        self.model_architecture = model_architecture
        self.dataset_name = dataset_name

    def initialize(self):
        seed = random.randint(0, 1000) if rand else None
        self.model = create_model(self.model_architecture, self.dataset_name, seed=seed)

    def compute_soft_labels(self, x_pub):
        with torch.no_grad():
            outputs = self.model(x_pub)  # compute outputs
            return torch.argmax(outputs, dim=1)  # return as same shape of labels

    def compress_soft_labels(self):
        pass  # todo compress

    def get_soft_labels(self):
        return self.soft_labels

    def train(self, train_loader):
        return train(self.model, train_loader)

    def fit(self):
        """Trains on local data and computes soft labels"""
        avg_loss, accuracy = self.train(self.train_loader)
        self.soft_labels = self.compute_soft_labels(self.x_pub)  # predict soft labels
        # compress here
        return avg_loss, accuracy
