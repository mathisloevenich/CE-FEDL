#!/usr/bin/env python3

import torch
import argparse
import random
import os
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
from data import cifar_data, femnist_data
from utils import evaluate
from client import DistillationClient
from strategy import DistillationStrategy


class Simulation:

    def __init__(
            self,
            num_clients: int,
            num_rounds: int,
            data_set_name: str,
            b_up=8,
            b_down=8,
            client_participation: float = 1.0
    ):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.data_set_name = data_set_name
        self.train_loaders, self.test_loaders = self.load_dataset(data_set_name)
        # don't need test_loaders so make public dataset by using it
        pub_trainloader = self.train_loaders[-1]  # last train set is public dataset
        x_pub = torch.cat([batch_x for batch_x, _ in pub_trainloader])
        y_pub = torch.cat([batch_y for _, batch_y in pub_trainloader])
        # print(x_pub.shape, y_pub.shape)
        self.strategy = DistillationStrategy(x_pub, y_pub, data_set_name)
        self.clients = [self.client_fn(cid, x_pub, data_set_name) for cid in range(num_clients)]
        num_samples = int(len(self.clients) * client_participation)
        self.participating_clients = random.sample(self.clients, num_samples)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_dataset(self, data_set):
        if data_set == "cifar":
            return cifar_data(self.num_clients + 1, balanced_data=True)  # get one more for public dataset
        elif data_set == "femnist":
            return femnist_data(combine_clients=self.num_clients + 1)

    def client_fn(self, cid, x_pub, model_name):
        train_loader = self.train_loaders[cid]
        return DistillationClient(cid, train_loader, x_pub, model_name)

    def aggregate_soft_labels(self):
        """Get soft labels from clients and convert to float to aggregate them. """
        sl_float = [client.get_soft_labels().float() for client in self.participating_clients]
        aggregated_soft_labels = torch.mean(torch.stack(sl_float), dim=0)
        return aggregated_soft_labels

    def run_simulation(self):

        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        for t in range(1, self.num_rounds + 1):
            print("Round: ", t)
            for client in tqdm(self.participating_clients, desc="Training clients"):
                client.initialize()  # initialize client model (new)

                # only train after first round and distill public model soft label information
                if t > 1:
                    client.train(
                        DataLoader(
                            TensorDataset(self.strategy.get_x_pub(), self.strategy.get_soft_labels()),
                            batch_size=32
                        )
                    )  # Distillation

                client.fit()  # trains on training data and computes soft labels

            self.strategy.set_soft_labels(self.aggregate_soft_labels())
            train_loss, train_accuracy = self.strategy.fit()  # trains and computes soft labels to distill
            # add some metrics to evaluate
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            # evaluate on random test_loader from test_loaders
            val_loss, val_accuracy = self.strategy.evaluate(self.test_loaders[random.randint(0, self.num_clients)])
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"Eval Loss: {val_loss:.4f}, Eval Accuracy: {val_accuracy:.4f}")

            # server.train(public_data, aggregated_labels)
            # server_soft_labels = server.compute_soft_labels(public_data)
            # server.compress_labels(server_soft_labels, bdown)

        self.save_data(train_losses, train_accuracies, val_losses, val_accuracies)

    def save_data(self, train_losses, train_accuracies, val_losses, val_accuracies):

        data = pd.DataFrame({
            'round': list(range(1, self.num_rounds + 1)),
            'train_loss': train_losses,
            'train_accuracy': train_accuracies,
            'val_loss': val_losses,
            'val_accuracies': val_accuracies
        })

        # Dateipfad
        directory = "results_data"

        index = 1
        while True:
            file_path = f"{directory}/{self.data_set_name}-cl{self.num_clients}-nr{self.num_rounds}_{index}.csv"
            if os.path.exists(file_path):
                index += 1
            else:
                data.to_csv(file_path, index=False)
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Compressed Federated Distillation Simulation")
    parser.add_argument("-n", "--num_clients", type=int, default=10, help="Number of clients")
    parser.add_argument("-r", "--num_rounds", type=int, default=5, help="Number of rounds")
    parser.add_argument("-d", "--data_set", type=str, default="cifar", help="Specify data set")
    parser.add_argument("--bup", type=int, default=8, help="Upstream precision")
    parser.add_argument("--bdown", type=int, default=8, help="Downstream precision")

    args = parser.parse_args()

    simulation = Simulation(args.num_clients, args.num_rounds, args.data_set, args.bup, args.bdown)

    print("Running on:", simulation.device)

    print("Run Simulation")
    simulation.run_simulation()
