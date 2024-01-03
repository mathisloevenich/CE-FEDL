import pickle
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import flwr as fl

from dataset import prepare_dataset, femnist_data, cifar_data
from client import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn, weighted_average

config_name="base_femnist" 
# A decorator for Hydra. This tells hydra to by default load the config in conf/base.yaml
@hydra.main(config_path="conf", config_name=config_name, version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    # Hydra automatically creates a directory for your experiments
    # by default it would be in <this directory>/outputs/<date>/<time>
    # you can retrieve the path to it as shown below. We'll use this path to
    # save the results of the simulation (see the last part of this main())
    save_path = HydraConfig.get().runtime.output_dir

    ## 2. Prepare your dataset
    if config_name=="base_cifar":
        trainloaders, validationloaders, testloader = prepare_dataset(num_partitions=cfg.num_clients, batch_size=cfg.batch_size, dataset_func=cifar_data)
    elif config_name=="base_femnist":
        trainloaders, validationloaders = femnist_data(batch_size=cfg.batch_size, combine_clients=1, subset=cfg.num_clients)

    ## 3. Define your clients
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    ## 4. Define your strategy
    if config_name=="base_cifar":
        strategy = fl.server.strategy.FedAvg(
                fraction_fit=0.0,  
                min_fit_clients=cfg.num_clients_per_round_fit, 
                fraction_evaluate=0.5,
                min_evaluate_clients=cfg.num_clients_per_round_eval,
                min_available_clients=cfg.num_clients,
                on_fit_config_fn=get_on_fit_config(
                    cfg.config_fit
                ),
                evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
                evaluate_metrics_aggregation_fn=weighted_average,
        )
        
    elif config_name=="base_femnist":
        strategy = fl.server.strategy.FedAvg(
                fraction_fit=0.0,
                min_fit_clients=cfg.num_clients_per_round_fit, 
                fraction_evaluate=0.0,
                min_evaluate_clients=cfg.num_clients_per_round_eval,
                min_available_clients=cfg.num_clients,
                on_fit_config_fn=get_on_fit_config(
                    cfg.config_fit
                ),
                evaluate_metrics_aggregation_fn=weighted_average,
        )

    ## 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(
            num_rounds=cfg.num_rounds
        ),
        strategy=strategy,
        client_resources={
            "num_cpus": 6,
            "num_gpus": 0.1,
        },
    )

    ## 6. Save your results
    results_path = Path(save_path) / "results.pkl"

    results = {"history": history}

    # save the results as a python pickle
    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()