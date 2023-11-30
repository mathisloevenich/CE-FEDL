import pickle
import os
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
import flwr as fl
from flwr.common import ndarrays_to_parameters
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset
from client import generate_client_fn, FlowerClient
from server import create_tracker_strategy


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    # Prepare dataset
    train_loaders, validation_loaders, test_loader = prepare_dataset(cfg.num_clients, cfg.batch_size)

    print(len(train_loaders), len(train_loaders[0].dataset))

    # Define clients
    client_fn = generate_client_fn(train_loaders, validation_loaders, cfg.num_classes)

    initial_parameters = ndarrays_to_parameters(client_fn("0").get_parameters(cfg))


    # Define strategy
    strategy = create_tracker_strategy(fl.server.strategy.FedAvgM, cfg, test_loader, initial_parameters)
    """
    strategy = fl.server.strategy.FedAvg(fraction_fit=0.00001,
                                         min_fit_clients=cfg.num_clients_per_round_fit,
                                         fraction_evaluate=0.00001,
                                         min_evaluate_clients=cfg.num_clients_per_round_eval,
                                         min_available_clients=cfg.num_clients,
                                         on_fit_config_fn=get_on_fit_config(cfg.config_fit),
                                         evaluate_fn=get_evaluate_fn(cfg.num_classes, test_loader))
    """

    # Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={'num_cpus': 6, 'num_gpus': 0.1}
    )

    bytes_sent = strategy.get_data_sent_per_round()
    bytes_received = strategy.get_data_received_per_round()

    # Save results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / 'results.pkl'
    results = {'history': history, 'server_sent': bytes_sent, 'server_received': bytes_received}

    with open(str(results_path), 'wb') as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    os.environ["RAY_DEDUP_LOGS"] = "0"
    main()

