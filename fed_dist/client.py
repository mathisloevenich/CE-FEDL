from torch.utils.data import DataLoader

from models import create_model
from utils import train, train_on_soft_labels, compute_soft_labels

import torch
import random


class DistillationClient:

    def __init__(
            self,
            cid,
            train_loader: DataLoader,
            x_pub,
            model_architecture="cnn500k",
            dataset_name="cifar",
            optimiser="Adam"
    ):
        self.cid = cid
        self.train_loader = train_loader
        self.x_pub = x_pub  # fixed public data
        self.soft_labels = None
        self.model = None
        self.model_architecture = model_architecture
        self.dataset_name = dataset_name
        self.optimiser = optimiser

    def initialize(self, rand=True):
        seed = random.randint(0, 1000) if rand else None
        self.model = create_model(self.model_architecture, self.dataset_name, seed=seed)

    def compress_soft_labels(self):
        pass  # todo compress

    def get_soft_labels(self):
        return self.soft_labels

    def train(self, train_loader, epochs=1, train_fn="train"):

        history = {
            "losses": [],
            "accuracies": []
        }

        methods = {
            "train": train,
            "train_sl": train_on_soft_labels
        }

        for epoch in range(epochs):
            loss, acc = methods[train_fn](self.model, train_loader, optimiser=self.optimiser)
            history["losses"].append(loss)
            history["accuracies"].append(acc)

        return sum(history["losses"]) / epochs, sum(history["accuracies"]) / epochs

    def fit(self, epochs=1):
        """Trains on local data and computes soft labels"""
        avg_loss, accuracy = self.train(self.train_loader, epochs)
        self.soft_labels = compute_soft_labels(self.model, self.x_pub)  # predict soft labels
        # compress here
        return avg_loss, accuracy
