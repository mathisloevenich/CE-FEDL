
import random

import numpy as np
import torch

from typing import List, Tuple, Union, Optional, Dict
import flwr as fl

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from torch.utils.data import DataLoader, TensorDataset

from utils import get_parameters, train_on_soft_labels, compute_soft_labels, parameters_to_tensor, tensor_to_parameters


class FedStrategy(FedAvg):
    def __init__(
        self,
        model,
        x_pub,
        conf,
    ) -> None:
        super().__init__()
        self.model = model
        self.x_pub = x_pub
        self.dist_parameters = torch.rand((len(self.x_pub), 32))  # server computed soft labels
        self.client_epochs = conf["client_epochs"]
        self.client_dist_epochs = conf["client_dist_epochs"]
        self.server_epochs = conf["server_epochs"]
        self.optimiser = conf["server_optimiser"]
        self.verbose = conf["verbose"]

        self.fraction_fit = conf["fraction_fit"]
        self.fraction_evaluate = conf["fraction_evaluate"]
        self.min_fit_clients = conf["min_fit_clients"]
        self.min_evaluate_clients = conf["min_evaluate_clients"]
        self.min_available_clients = conf["min_available_clients"]

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        ndarrays = get_parameters(self.model)
        return ndarrays_to_parameters(ndarrays)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        random.seed(server_round)
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients
        )

        # return the sampled clients
        fit_ins = FitIns(
            parameters, {
                "client_epochs": self.client_epochs,
                "server_round": server_round,
                "client_dist_epochs": self.client_dist_epochs,
                "verbose": self.verbose
            }
        )
        return [(client, fit_ins) for client in clients]

    def train(self, epochs=1, verbose=False):
        print(f"Train Server")
        history = {
            "losses": [],
            "accuracies": []
        }

        public_loader = DataLoader(
            TensorDataset(self.x_pub, self.dist_parameters),
            batch_size=32
        )
        for epoch in range(epochs):
            epoch_loss, epoch_acc = train_on_soft_labels(self.model, public_loader, optimiser=self.optimiser)
            history["losses"].append(epoch_loss)
            history["accuracies"].append(epoch_acc)

            if verbose:
                print(f"Epoch {epoch + 1}: server train loss {epoch_loss}, accuracy {epoch_acc}")

        return sum(history["losses"]) / epochs, sum(history["accuracies"]) / epochs

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        # # aggregate soft labels
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # set soft labels
        self.dist_parameters = parameters_to_tensor(parameters_aggregated)

        loss, accuracy = self.train(self.server_epochs, self.verbose)  # train on soft labels

        # compute new soft labels
        new_soft_labels = tensor_to_parameters(compute_soft_labels(self.model, self.x_pub))

        # send parameters to clients
        return new_soft_labels, {"accuracy": float(accuracy)}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []

        # Evaluation is done with weights
        evaluate_ins = EvaluateIns(
            get_parameters(self.model), {
                "verbose": self.verbose
            }
        )

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}

        weighted_metrics = [
            (evaluate_res.num_examples,
             evaluate_res.num_examples * evaluate_res.loss,
             evaluate_res.num_examples * evaluate_res.metrics["accuracy"]) for _, evaluate_res in results
        ]
        total_num_examples = sum([value[0] for value in weighted_metrics])
        aggregated_loss = sum([value[1] for value in weighted_metrics]) / total_num_examples
        aggregated_metrics = {"accuracy": sum([value[2] for value in weighted_metrics]) / total_num_examples}

        return aggregated_loss, aggregated_metrics

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None

        # We deserialize using our custom method
        parameters_ndarrays = parameters_to_ndarrays(parameters)

        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
