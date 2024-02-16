import argparse
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


class Simulation:

    def __init__(self, conf):
        self.num_clients = conf["num_clients"]
        self.client_participation = conf["client_participation"]
        self.client_epochs = conf["client_epochs"]
        self.client_dist_epochs = conf["client_epochs"]
        self.client_architecture = conf["client_model"]
        self.client_optimiser = conf["client_optimiser"]
        self.num_rounds = conf["num_rounds"]
        self.server_epochs = conf["server_epochs"]
        self.server_architecture = conf["server_model"]
        self.server_optimiser = conf["server_optimiser"]
        self.dataset_name = conf["data_set"]
        self.public_data_size = conf["public_data_size"]

        # load dataset
        self.train_loaders, self.val_loaders, self.public_loader = self.load_dataset(
            self.dataset_name,
            public_data_size=self.public_data_size
        )
        self.public_data = torch.cat([batch_x for batch_x in self.public_loader])
        # self.public_labels = torch.cat([batch_y for _, batch_y in self.public_loader])

        server_model = create_model(self.server_architecture, self.dataset_name)  # create server model
        self.strategy = FedStrategy(server_model, self.public_data, self.server_optimiser)

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

    def load_dataset(self, data_set, public_data_size=10000):
        if data_set == "cifar":
            return cifar_data(
                self.num_clients,
                balanced_data=True,
                public_data_size=public_data_size
            )  # get one more for public dataset
        elif data_set == "femnist":
            return femnist_data(combine_clients=self.num_clients)

    def run_simulation(self):
        client_resources = None
        if DEVICE == "cuda":
            client_resources = {"num_gpus": 1, "num_cpus": 1}

        history = fl.simulation.start_simulation(
            client_fn=self.client_fn,
            num_clients=self.num_clients,
            config=fl.server.ServerConfig(num_rounds=3),
            strategy=self.strategy,
            client_resources=client_resources
        )

        return history


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
        "client_epochs": 8,
        "client_dist_epochs": 2,
        "client_participation": 1.0,
        "client_optimiser": "Adam",
        "client_model": "resnet18",
        "server_epochs": 10,
        "server_optimiser": "Adam",
        "server_model": "resnet18",
        "public_data_size": 1000
    }

    simulation = Simulation(config)

    print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")
    print("Public data shape:", simulation.public_data.shape)

    # start calculation runtime
    start = datetime.now()
    history = simulation.run_simulation()
    end = datetime.now()

    total_runtime = end - start
    print("Total runtime: ", total_runtime)

    # results = pd.DataFrame({
    #     "date": [datetime.now()],
    #     "runtime": [end-start],
    #     "dataset": [dataset_name],
    #     "frac_clients": [frac_clients],
    #     "num_clients": [num_clients],
    #     "num_rounds": [num_rounds],
    #     "approach": [approach],
    #     "epochs": [epochs],
    #     "sparsify_by": [sparsify_by],
    #     "keep_first_last": [keep_first_last],
    #     "learning_rate": [learning_rate],
    #     "regularisation": [regularisation],
    #     "losses": [history.losses_distributed],
    #     "accs": [history.metrics_distributed["accuracy"]]
    # })
