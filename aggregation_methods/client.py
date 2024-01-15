from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar

import numpy as np
import torch
import flwr as fl

from model import get_model, train, test


class FlowerClient(fl.client.NumPyClient):
    """Define a Flower Client."""

    def __init__(self, trainloader, vallodaer, num_classes) -> None:
        super().__init__()

        # the dataloaders that point to the data associated to this client
        self.trainloader = trainloader
        self.valloader = vallodaer

        # a model that is randomly initialised at first
        self.model = get_model(num_classes)

        # figure out if this client has access to GPU support or not
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Train model received by the server (parameters) using the data.

        that belongs to this client. Then, send it back to the server.
        """

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]

        # a very standard looking optimiser
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        train(self.model, self.trainloader, optim, epochs, self.device)

        return self.get_parameters({}), len(self.trainloader), {}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader), {"accuracy": accuracy}


def generate_client_fn(trainloaders, valloaders, num_classes):
    """Return a function that can be used by the VirtualClientEngine.

    to spawn a FlowerClient with client id `cid`.
    """

    def client_fn(cid: str):
        
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            vallodaer=valloaders[int(cid)],
            num_classes=num_classes,
        )

    # return the function to spawn client
    return client_fn