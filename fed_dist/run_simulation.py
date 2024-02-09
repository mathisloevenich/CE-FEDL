#!/usr/bin/env python3

import torch
import argparse
import random
import os
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
from utils import DEVICE
from data import cifar_data, femnist_data
from models import create_model
from client import DistillationClient
from strategy import DistillationStrategy


class Simulation:

    def __init__(self, conf):
        self.num_clients = conf["num_clients"]
        self.client_participation = conf["client_participation"]
        self.client_epochs = conf["client_epochs"]
        self.client_architecture = conf["client_model"]
        self.num_rounds = conf["num_rounds"]
        self.server_epochs = conf["server_epochs"]
        self.server_architecture = conf["server_model"]
        self.dataset_name = conf["data_set"]

        # load dataset
        self.train_loaders, self.test_loader, self.public_loader = self.load_dataset(self.dataset_name)
        self.public_data = torch.cat([batch_x for batch_x, _ in self.public_loader])
        self.public_labels = torch.cat([batch_y for _, batch_y in self.public_loader])

        sever_model = create_model(self.server_architecture, self.dataset_name)  # create server model
        self.strategy = DistillationStrategy(sever_model, self.public_data)
        self.clients = [
            self.client_fn(cid, self.public_data, self.client_architecture)
            for cid in range(self.num_clients)
        ]
        num_samples = int(len(self.clients) * self.client_participation)
        self.participating_clients = random.sample(self.clients, num_samples)

    def load_dataset(self, data_set):
        if data_set == "cifar":
            return cifar_data(self.num_clients, balanced_data=True)  # get one more for public dataset
        elif data_set == "femnist":
            return femnist_data(combine_clients=self.num_clients)

    def client_fn(self, cid, x_pub, model_architecture):
        train_loader = self.train_loaders[cid]
        return DistillationClient(cid, train_loader, x_pub, model_architecture, self.dataset_name)

    def aggregate_soft_labels(self):
        """Get soft labels from clients and convert to float to aggregate them. """
        soft_labels = [client.get_soft_labels() for client in self.participating_clients]
        return torch.mean(torch.stack(soft_labels), dim=0)

    def run_simulation(self):

        history = {
            "train_losses": [],
            "train_accuracies": [],
            "val_losses": [],
            "val_accuracies": []
        }

        for t in range(1, self.num_rounds + 1):
            print("Round: ", t)

            # only allocate once for every client
            client_loader = DataLoader(
                TensorDataset(self.strategy.get_x_pub(), self.strategy.get_soft_labels()),
                batch_size=32
            )

            for client in tqdm(self.participating_clients, desc="Training clients"):
                client.initialize()  # initialize client model (new)

                # only train after first round and distill public model soft label information
                if t > 1:
                    client.train(client_loader, train_fn="train_sl")  # Distillation

                client.fit(epochs=self.client_epochs)  # trains on training data and computes soft labels

            self.strategy.set_soft_labels(self.aggregate_soft_labels())
            train_loss, train_accuracy = self.strategy.fit(epochs=self.server_epochs)

            # add some metrics to evaluate
            history["train_losses"].append(train_loss)
            history["train_accuracies"].append(train_accuracy)

            # evaluate on random test_loader from test_loaders
            val_loss, val_accuracy = self.strategy.evaluate(self.test_loader)
            history["val_losses"].append(val_loss)
            history["val_accuracies"].append(val_accuracy)

            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"Eval Loss: {val_loss:.4f}, Eval Accuracy: {val_accuracy:.4f}")

        self.save_data(self.strategy.model, history)

    def save_data(self, model, history):

        data = pd.DataFrame({
            'round': list(range(1, self.num_rounds + 1)),
            'train_loss': history["train_losses"],
            'train_accuracy': history["train_accuracies"],
            'val_loss': history["val_losses"],
            'val_accuracies': history["val_accuracies"]
        })

        # Dateipfad
        directory = "results_data"

        path_prefix = (
            f"{directory}/{self.server_architecture}-{self.dataset_name}-cl{self.num_clients}-nr{self.num_rounds}"
        )

        index = 1
        while os.path.exists(f"{path_prefix}_{index}.csv"):
            index += 1

        data.to_csv(f"{path_prefix}_{index}.csv", index=False)

        torch.save(model.state_dict(), f"{path_prefix}_{index}_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Compressed Federated Distillation Simulation")
    parser.add_argument("-n", "--num_clients", type=int, default=10, help="Number of clients")
    parser.add_argument("-r", "--num_rounds", type=int, default=5, help="Number of rounds")
    parser.add_argument("-d", "--data_set", type=str, default="cifar", help="Specify data set")
    parser.add_argument("--bup", type=int, default=8, help="Upstream precision")
    parser.add_argument("--bdown", type=int, default=8, help="Downstream precision")

    args = parser.parse_args()

    config = {
        "num_clients": args.num_clients,
        "num_rounds": args.num_rounds,
        "data_set": args.data_set,
        "client_epochs": 1,
        "client_participation": 1.0,
        "client_optimiser": "Adam",
        "client_model": "cnn500k",
        "server_epochs": 1,
        "server_optimiser": "Adam",
        "server_model": "cnn500k",
    }

    simulation = Simulation(config)

    print("Running on:", DEVICE)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print("Public data shape:", simulation.public_data.shape)
    print("Public labels shape:", simulation.public_labels.shape)
    simulation.run_simulation()
