# Code by Natasha
# Last updated: 2023.12.30

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torch.utils.data import DataLoader, random_split
from fedlab.utils.dataset.partition import CIFAR10Partitioner
import json
import numpy as np
from tqdm import tqdm
from io import BytesIO
from flwr.common import Parameters

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_on_soft_labels(
        model,
        train_loader,
        optimiser="SGD",
        lr=0.1,
        weight_decay=0,
):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    # criterion = nn.KLDivLoss(reduction='batchmean').to(DEVICE)
    optim_dict = {"SGD": torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay),
                  "Adam": torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)}
    optimiser = optim_dict[optimiser]

    metrics = {
        "batch_loss": 0.0,
        "batch_correct": 0.0,
        "batch_samples": 0.0
    }

    # set model on train mode
    model.train()

    # for each mini-batch
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimiser.zero_grad()

        # forward pass
        logits = model(inputs)

        loss = criterion(logits, targets)

        predicted_labels = torch.argmax(logits, dim=1)
        true_labels = torch.argmax(targets, dim=1)

        # Update metrics
        metrics["batch_correct"] += (predicted_labels == true_labels).sum().item()
        metrics["batch_loss"] += loss.item()
        metrics["batch_samples"] += targets.size(0)

        # Bachward pass and optimize
        loss.backward()
        optimiser.step()

    avg_loss = metrics["batch_loss"] / len(train_loader)
    avg_acc = metrics["batch_correct"] / metrics["batch_samples"]
    return avg_loss, avg_acc


def train(model,
          train_loader,
          optimiser="SGD",
          lr=0.01,
          weight_decay=0,
          ):

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optim_dict = {"SGD": torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay),
                  "Adam": torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)}
    optimiser = optim_dict[optimiser]

    metrics = {
        "batch_loss": 0.0,
        "batch_correct": 0.0,
        "batch_samples": 0.0
    }

    # set model on train mode
    model.train()

    # for each mini-batch
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimiser.zero_grad()

        # Forward pass
        logits = model(inputs)
        loss = criterion(logits, targets)

        # Backward pass and optimize
        loss.backward()
        optimiser.step()

        predicted = torch.argmax(logits, dim=1)

        # Update metrics
        metrics["batch_correct"] += (predicted == targets).sum().item()
        metrics["batch_loss"] += loss.item()
        metrics["batch_samples"] += targets.size(0)

    avg_loss = metrics["batch_loss"] / len(train_loader)
    avg_acc = metrics["batch_correct"] / metrics["batch_samples"]
    return avg_loss, avg_acc


def evaluate(model, dataloader):
    model.eval()

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss().to(DEVICE)

    metrics = {
        "eval_loss": 0.0,
        "eval_correct": 0.0,
        "eval_samples": 0.0
    }

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Forward pass
            logits = model(inputs)
            loss = criterion(logits, targets)

            metrics["eval_loss"] += loss.item()
            predicted = torch.argmax(logits, dim=1)
            metrics["eval_correct"] += (predicted == targets).sum().item()
            metrics["eval_samples"] += targets.size(0)

    avg_loss = metrics["eval_loss"] / len(dataloader)
    accuracy = metrics["eval_correct"] / metrics["eval_samples"]
    return avg_loss, accuracy


def predict(model, inputs):
    # do not allocate gradients to memory
    with torch.no_grad():
        model, inputs = model.to(DEVICE), inputs.to(DEVICE)
        return model(inputs)


def compute_soft_labels(model, inputs):
    logits = predict(model, inputs)
    return F.softmax(logits, dim=1)
