# Code by Natasha
# Last updated: 2023.12.30

import torch
from torch import nn

from torchvision.models import resnet18


def create_model(dataset_name, seed=47):
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


def get_resnet18_femnist(seed=47):
    torch.manual_seed(seed)
    femnist_model = resnet18(num_classes=62)
    femnist_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    return femnist_model

def get_resnet18_cifar10(seed=47):
    torch.manual_seed(seed)
    cifar10_model = resnet18(num_classes=10)

    return cifar10_model