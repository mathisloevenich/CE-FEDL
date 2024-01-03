import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import numpy as np

def get_resnet18(num_classes):
    if num_classes==10:
        return get_resnet18_cifar10()
    elif num_classes==62:
        return get_resnet18_femnist()
    else:
        return resnet18(num_classes=num_classes)

def get_resnet18_femnist(seed=47):
    torch.manual_seed(seed)
    femnist_model = resnet18(num_classes=62)
    femnist_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    return femnist_model

def get_resnet18_cifar10(seed=47):
    torch.manual_seed(seed)
    cifar10_model = resnet18(num_classes=10)

    return cifar10_model

def train(net, trainloader, optimizer, epochs, device: str):
    """Train the network on the training set.

    This is a fairly simple training loop for PyTorch.
    """
    criterion = nn.CrossEntropyLoss()

    net.train()
    net.to(device)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader, device: str):
    """Validate the network on the entire test set.

    and report loss and accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
