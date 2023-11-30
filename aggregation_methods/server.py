import flwr.server.strategy
from flwr.common import Parameters, Scalar, EvaluateRes, EvaluateIns, FitRes, FitIns
from flwr.server.client_proxy import ClientProxy
from typing import Optional, Tuple, Dict, List, Union
from pympler import asizeof

import torch
import pickle
import flwr as fl

from flwr.server.strategy import FedAvgM, FedAvg
from omegaconf import DictConfig
from model import Net, test
from collections import OrderedDict


def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):

        return {'lr': config.lr,
                'momentum': config.momentum,
                'local_epochs': config.local_epochs}

    return fit_config_fn


def get_evaluate_fn(num_classes: int, test_loader):

    def evaluate_fn(server_round: int, parameters, config):

        model = Net(num_classes)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(model, test_loader, device)

        return loss, {"accuracy": accuracy}

    return evaluate_fn


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
