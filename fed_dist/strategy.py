import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import train_on_soft_labels, evaluate, compute_soft_labels


class DistillationStrategy:

    def __init__(self, model, x_pub):

        # take resnet models
        self.model = model
        self.x_pub = x_pub  # fixed dataset
        self.y_pub_distill = None  # server computed soft labels

    def get_x_pub(self):
        return self.x_pub

    def set_soft_labels(self, new_y_pub):
        self.y_pub_distill = new_y_pub

    def get_soft_labels(self):
        return self.y_pub_distill

    def train(self):
        return train_on_soft_labels(
            self.model,
            DataLoader(TensorDataset(self.x_pub, self.y_pub_distill), batch_size=32)
        )

    def fit(self):
        """Trains on x_pub and y_pub, computes new soft labels to distill to the clients"""
        avg_loss, accuracy = self.train()
        self.y_pub_distill = self.compute_soft_labels()
        return avg_loss, accuracy

    def evaluate(self, test_loader):
        return evaluate(self.model, test_loader)


