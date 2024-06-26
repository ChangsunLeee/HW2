import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MNIST
from model import LeNet5, CustomMLP
import matplotlib.pyplot as plt

def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    for data, target in trn_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    trn_loss = total_loss / len(trn_loader.dataset)
    acc = correct / len(trn_loader.dataset)
    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in tst_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    tst_loss = total_loss / len(tst_loader.dataset)
    acc = correct / len(tst_loader.dataset)
    return tst_loss, acc

def plot_statistics(train_losses, train_accuracies, test_losses, test_accuracies, model_name):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve - {model_name}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Curve - {model_name}')
    plt.legend()

    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MNIST(data_file="train.tar")
    test_dataset = MNIST(data_file="test.tar")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    models = {'LeNet-5': LeNet5(), 'Custom MLP': CustomMLP()}  
    criterions = {'LeNet-5': nn.CrossEntropyLoss(), 'Custom MLP': nn.CrossEntropyLoss()}
    optimizers = {'LeNet-5': optim.SGD(models['LeNet-5'].parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5),
                  'Custom MLP': optim.SGD(models['Custom MLP'].parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)}

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.to(device)

        criterion = criterions[model_name]
        optimizer = optimizers[model_name]

        train_losses, train_accuracies = [], []
        test_losses, test_accuracies = [], []

        for epoch in range(1, 11): 
            trn_loss, trn_acc = train(model, train_loader, device, criterion, optimizer)
            tst_loss, tst_acc = test(model, test_loader, device, criterion)
            print(f"Epoch {epoch}:")
            print(f"Train Loss: {trn_loss:.4f}, Accuracy: {trn_acc:.4f}")
            print(f"Test Loss: {tst_loss:.4f}, Accuracy: {tst_acc:.4f}")

            train_losses.append(trn_loss)
            train_accuracies.append(trn_acc)
            test_losses.append(tst_loss)
            test_accuracies.append(tst_acc)

        plot_statistics(train_losses, train_accuracies, test_losses, test_accuracies, model_name)

if __name__ == '__main__':
    main()
