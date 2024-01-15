import torch
import flwr as fl
from flwr.common import Parameters, Scalar, EvaluateRes, EvaluateIns, FitRes, FitIns
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvgM, FedAvg
import numpy as np

from typing import Optional, Tuple, Dict, List, Union
from pympler import asizeof

from collections import OrderedDict
from omegaconf import DictConfig

from model import get_model, test


def set_weights(model: torch.nn.ModuleList, weights: fl.common.NDArrays) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)

    
def get_weights(model: torch.nn.ModuleList) -> fl.common.NDArrays:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def get_on_fit_config(config: DictConfig):
    """Return function that prepares config to send to clients."""

    def fit_config_fn(server_round: int):

        return {
            "lr": config.lr,
            "momentum": config.momentum,
            "local_epochs": config.local_epochs,
        }

    return fit_config_fn


def get_evaluate_fn(num_classes: int, testloader):
    """Define function for global evaluation on the server."""

    def evaluate_fn(server_round: int, parameters, config):

        model = get_model(num_classes)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(model, testloader, device)

        return loss, {"accuracy": accuracy}

    return evaluate_fn

def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def create_tracker_strategy(server_strategy, cfg, test_loader, initial_parameters):

    class ComTracker(server_strategy):

        def __init__(self):
            if server_strategy == FedAvgM:
                super().__init__(
                    fraction_fit=0.00001,
                    min_fit_clients=cfg.num_clients_per_round_fit,
                    fraction_evaluate=0.00001,
                    min_evaluate_clients=cfg.num_clients_per_round_eval,
                    min_available_clients=cfg.num_clients,
                    on_fit_config_fn=get_on_fit_config(cfg.config_fit),
                    evaluate_fn=get_evaluate_fn(cfg.num_classes, test_loader),
                    server_momentum=cfg.server_momentum,
                    initial_parameters=initial_parameters
                )
            else:
                super().__init__(
                    fraction_fit=0.00001,
                    min_fit_clients=cfg.num_clients_per_round_fit,
                    fraction_evaluate=0.00001,
                    min_evaluate_clients=cfg.num_clients_per_round_eval,
                    min_available_clients=cfg.num_clients,
                    on_fit_config_fn=get_on_fit_config(cfg.config_fit),
                    evaluate_fn=get_evaluate_fn(cfg.num_classes, test_loader)
                )
            self.data_sent_per_round = []
            self.data_received_per_round = []

        def aggregate_fit(self,
                          server_round: int,
                          results: List[Tuple[ClientProxy, FitRes]],
                          failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
                          ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            # Track data received from clients
            total_received_this_round = asizeof.asizeof(results)
            self.data_received_per_round.append(total_received_this_round)

            aggregated = super().aggregate_fit(server_round, results, failures)
            if aggregated is not None:
                parameters, _ = aggregated
                data_size = asizeof.asizeof(parameters)
                self.data_sent_per_round.append(data_size)
                print(f"Round {server_round}: Data size sent to clients: {data_size} bytes")

            return aggregated

        def get_data_sent_per_round(self):
            return self.data_sent_per_round

        def get_data_received_per_round(self):
            return self.data_received_per_round

    return ComTracker()