import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import numpy as np


def create_model(dataset_name):
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

    torch.manual_seed(47)
    return CNN500k(num_channels, image_size, num_classes)


class CNN500k(nn.Module):
    def __init__(self, num_channels, image_size, num_classes):
        super().__init__()

        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(2, 2)
        self.drop1 = torch.nn.Dropout(0.5)
        self.padding = torch.nn.ReplicationPad2d(1)
        
        self.c1 = torch.nn.Conv2d(num_channels, 32, kernel_size=3, padding=1, bias=False)
        self.c2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.c3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.c4 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)

        self.fc1 = torch.nn.Linear(32 * int(image_size/8) * int(image_size/8), 512, bias=False)
        self.fc2 = torch.nn.Linear(512, 256, bias=False)
        self.fc4 = torch.nn.Linear(256, num_classes, bias=False)

    def forward(self, x):
        
        h_list=[]
        h_list.append(torch.mean(x, 0, True))
        con1 = self.relu(self.c1(x))

        h_list.append(torch.mean(con1, 0, True))
        con2 = self.relu(self.c2(con1))
        con2_p = self.maxpool(con2)

        h_list.append(torch.mean(con2_p, 0, True))
        con3 = self.relu(self.c3(con2_p))
        con3_p = self.maxpool(con3)

        h_list.append(torch.mean(con3_p, 0, True))
        con4 = self.relu(self.c4(con3_p))
        con4_p = self.maxpool(con4)

        h = con4_p.view(x.size(0), -1)
        h_list.append(torch.mean(h, 0, True))
        h = self.drop1(self.relu(self.fc1(h)))

        h_list.append(torch.mean(h, 0, True))
        h = self.drop1(self.relu(self.fc2(h)))

        h_list.append(torch.mean(h, 0, True))
        y = self.fc4(h)

        return y
        

def get_model(num_classes):
    if num_classes==62:
        num_channels=1
        image_size=28
    elif num_classes==10:
        num_channels=3
        image_size=32

    torch.manual_seed(47)
    return CNN500k(num_channels, image_size, num_classes)

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
