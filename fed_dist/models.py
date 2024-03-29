# Code by Natasha
# Last updated: 2023.12.30

import torch
from torch import nn
import torchvision.models as models

from utils import DEVICE

SEED = 42


def create_model(architecture, dataset_name, seed=SEED):
    if seed is None:
        seed = SEED

    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)

    if architecture == "cnn500k":
        return create_cnn500k(dataset_name, seed)
    if architecture == "resnet18":
        if dataset_name == "femnist":
            return models.resnet18(num_classes=62)  # 62 classes in FEMNIST dataset
        elif dataset_name == "cifar":
            return models.resnet18(num_classes=10)  # 10 classes in CIFAR dataset


def create_cnn500k(dataset_name, seed=SEED):
    """
    Input: Dataset name: can be 'femnist' or 'cifar'
    """
    if dataset_name=="femnist":
        num_channels=1
        image_size=28
        num_classes=62
    elif dataset_name=="cifar":
        num_channels=3
        image_size=32
        num_classes=10

    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
    return CNN500k(num_channels, image_size, num_classes)


class CNN500k(nn.Module):
    def __init__(self, num_channels, image_size, num_classes):
        super().__init__()
        
        self.layer_stack = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        
            nn.Flatten(),
            nn.Linear(32 * int(image_size/8) * int(image_size/8), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.layer_stack(x)
