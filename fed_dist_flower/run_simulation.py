import argparse
import json
import os
from datetime import datetime

import pandas as pd
import torch
import random

import flwr as fl

from utils import DEVICE
from data import cifar_data, femnist_data
from models import create_model
from client import FlowerClient
from strategy import FedStrategy

import logging
logging.basicConfig(level=logging.DEBUG)


class Simulation:

    def __init__(self, conf):
        self.num_clients = conf["num_clients"]
        self.client_epochs = conf["client_epochs"]
        self.client_dist_epochs = conf["client_epochs"]
        self.client_architecture = conf["client_model"]
        self.client_optimiser = conf["client_optimiser"]
        self.num_rounds = conf["num_rounds"]
        self.server_epochs = conf["server_epochs"]
        self.server_architecture = conf["server_model"]
        self.server_optimiser = conf["server_optimiser"]
        self.dataset_name = conf["data_set"]
        self.public_ratio = conf["public_ratio"]

        strategy_conf = {
            **conf,
            "fraction_fit": conf["client_participation"],
            "fraction_evaluate": 1.0,
            "min_fit_clients": 2,
            "min_evaluate_clients": 2,
            "min_available_clients": 2
        }

        self.conf = conf

        # load dataset
        self.train_loaders, self.val_loaders, self.public_loader = self.load_dataset(
            self.dataset_name,
            public_ratio=self.public_ratio
        )

        self.public_data = torch.cat([batch_x[0] for batch_x in self.public_loader], dim=0)

        server_model = create_model(self.server_architecture, self.dataset_name)  # create server model
        self.strategy = FedStrategy(
            server_model,
            self.public_data,
            strategy_conf
        )

    def client_fn(self, cid):
        train_loader = self.train_loaders[int(cid)]
        val_loader = self.val_loaders[int(cid)]
        return FlowerClient(
            cid,
            train_loader,
            val_loader,
            self.public_data,
            self.client_architecture,
            self.dataset_name,
            optimiser=self.client_optimiser
        )

    def load_dataset(self, data_set, public_ratio=10000):
        if data_set == "cifar":
            return cifar_data(
                self.num_clients,
                balanced_data=True,
                public_ratio=public_ratio
            )  # get one more for public dataset
        elif data_set == "femnist":
            return femnist_data(combine_clients=self.num_clients)

    def run_simulation(self):
        client_resources = None
        if DEVICE == "cuda":
            client_resources = {"num_gpus": 0.9, "num_cpus": 8}

        sim_hist = fl.simulation.start_simulation(
            client_fn=self.client_fn,
            num_clients=self.num_clients,
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=self.strategy,
            client_resources=client_resources
        )

        _, losses = zip(*sim_hist.losses_distributed)
        _, accuracies = zip(*sim_hist.metrics_distributed["accuracy"])
        _, train_accuracies = zip(*sim_hist.metrics_distributed_fit["accuracy"])

        hist = {
            "round": list(range(1, self.num_rounds + 1)),
            "losses": losses,
            "accuracies": accuracies,
            "train_accuracies": train_accuracies
        }

        # save data
        simulation.save_data(hist)

    def save_data(self, history):

        data = pd.DataFrame(history)

        # Dateipfad
        directory = "results_data"

        path_prefix = (
            f"{directory}/{self.server_architecture}-{self.dataset_name}-cl{self.num_clients}-nr{self.num_rounds}"
        )

        index = 1
        while os.path.exists(f"{path_prefix}_{index}.csv"):
            index += 1

        # store model results
        data.to_csv(f"{path_prefix}_{index}.csv", index=False)
        # store model
        torch.save(self.strategy.model.state_dict(), f"{path_prefix}_{index}_model.pth")
        # store config
        with open(f"{path_prefix}_{index}_conf.json", "w") as file:
            json.dump(self.conf, file, indent=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Compressed Federated Distillation Simulation")
    parser.add_argument("-n", "--num_clients", type=int, default=10, help="Number of clients")
    parser.add_argument("-r", "--num_rounds", type=int, default=5, help="Number of rounds")
    parser.add_argument("-d", "--data_set", type=str, default="cifar", help="Specify data set")
    parser.add_argument("--bup", type=int, default=8, help="Upstream precision")
    parser.add_argument("--bdown", type=int, default=8, help="Downstream precision")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")

    args = parser.parse_args()

    config = {
        "num_clients": args.num_clients,
        "num_rounds": args.num_rounds,
        "data_set": args.data_set,
        "verbose": args.verbose,
        "client_epochs": 1,
        "client_dist_epochs": 1,
        "client_participation": 1.0,
        "client_optimiser": "Adam",
        "client_model": "resnet18",
        "server_epochs": 10,
        "server_optimiser": "Adam",
        "server_model": "resnet18",
        "public_ratio": 0.5,
    }

    simulation = Simulation(config)

    print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")
    print("Public data shape:", simulation.public_data.shape)

    # start calculation runtime
    start = datetime.now()
    simulation.run_simulation()
    end = datetime.now()

    total_runtime = end - start
    print("Total runtime: ", total_runtime)

