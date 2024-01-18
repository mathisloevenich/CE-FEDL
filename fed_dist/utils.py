
import torch
import torch.nn as nn


# borrowed from Pytorch quickstart example
def train(model,
          train_loader,
          lr=0.1,
          weight_decay=0,
          optimiser="SGD",
          device="cpu"):
    """Train the network on the training set."""
    optim_dict = {"SGD": torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay),
                  "Adam": torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)}
    optimiser = optim_dict[optimiser]
    criterion = torch.nn.CrossEntropyLoss()

    model.train()  # Setzt das Modell in den Trainingsmodus

    total_loss, total_correct, total_samples = 0, 0, 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device).long()  # convert to long because of CEL
        optimiser.zero_grad()  # puts gradients to 0
        outputs = model(inputs)
        loss = criterion(outputs, targets)  # calculate loss
        loss.backward()  # calculate gradients
        optimiser.step()  # update the weights

        total_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        total_correct += predicted.eq(targets).sum().item()
        total_samples += targets.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def evaluate(model, data_loader, device="cpu"):
    model.eval()  # Setzt das Modell in den Bewertungsmodus

    criterion = torch.nn.CrossEntropyLoss()

    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():  # Deaktiviert die Gradientenberechnung
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


