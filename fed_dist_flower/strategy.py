
import random
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

from utils import get_parameters, train_on_soft_labels, compute_soft_labels


class FedStrategy(FedAvg):
    def __init__(
        self,
        model,
        x_pub,
        optimiser="Adam",
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ) -> None:
        super().__init__()
        self.model = model
        self.x_pub = x_pub
        self.y_pub_distill = torch.rand((len(self.x_pub), 32))  # server computed soft labels
        self.client_epochs = 1
        self.client_dist_epochs = 1
        self.server_epochs = 1
        self.optimiser = optimiser

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

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
                "epochs": self.client_epochs,
                "server_round": server_round,
                "client_dist_epochs": self.client_dist_epochs
            }
        )
        return [(client, fit_ins) for client in clients]

    def train(self, epochs=1):

        history = {
            "losses": [],
            "accuracies": []
        }

        public_loader = DataLoader(
            TensorDataset(self.x_pub, self.y_pub_distill),
            batch_size=32
        )
        for epoch in range(epochs):
            loss, acc = train_on_soft_labels(self.model, public_loader, optimiser=self.optimiser)
            history["losses"].append(loss)
            history["accuracies"].append(acc)

        return sum(history["losses"]) / epochs, sum(history["accuracies"]) / epochs

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        # aggregate soft labels
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # soft_labels = [client.get_soft_labels() for client in self.participating_clients]
        # torch.mean(torch.stack(soft_labels, dim=0), dim=0)

        # set soft labels
        self.y_pub_distill = parameters_aggregated
        loss, accuracy = self.train(epochs=self.server_epochs)  # train on soft labels

        # compute new soft labels
        new_soft_labels = compute_soft_labels(self.model, self.x_pub)

        return new_soft_labels, {"accuracy": float(accuracy)}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

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

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""

        # Let's assume we won't perform the global model evaluation on the server side.
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients