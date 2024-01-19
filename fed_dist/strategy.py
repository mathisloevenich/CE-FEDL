import torch
from torch.utils.data import DataLoader, TensorDataset

from models import get_resnet18_cifar10, get_resnet18_femnist
from utils import train, evaluate


class DistillationStrategy:

    def __init__(self, x_pub, y_pub, model_name="cifar"):

        # take resnet models
        self.model = get_resnet18_cifar10() if model_name == "cifar" else get_resnet18_femnist()
        self.x_pub = x_pub  # fixed dataset
        self.y_pub = None  # non-fixed soft-labels
        self.y_pub_distill = y_pub  # server computed soft labels

    def get_x_pub(self):
        return self.x_pub

    def set_soft_labels(self, new_y_pub):
        self.y_pub = new_y_pub

    def get_soft_labels(self):
        return self.y_pub_distill

    def compute_soft_labels(self):
        with torch.no_grad():
            outputs = self.model(self.x_pub)  # compute outputs
            return torch.argmax(outputs, dim=1)  # return as same shape of labels

    def train(self):
        return train(self.model, DataLoader(TensorDataset(self.x_pub, self.y_pub), batch_size=32))

    def fit(self):
        """Trains on x_pub and y_pub, computes new soft labels to distill to the clients"""
        avg_loss, accuracy = self.train()
        self.y_pub_distill = self.compute_soft_labels()
        return avg_loss, accuracy

    def evaluate(self, test_loader):
        return evaluate(self.model, test_loader)

