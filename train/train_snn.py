import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data.load_dataset import load_data
from models.snn_model import SpikingCNN
from utils.checkpoint import save_checkpoint


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    model = SpikingCNN(T=10, slope=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loader, test_loader = load_data(batch_size=64)

    epochs = 10
    best_test_acc = 0

    for epoch in range(epochs):

        model.train()
        running_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for images, labels in loop:

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)

        print(f"\nEpoch [{epoch+1}/{epochs}] "
              f"Loss: {running_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}% "
              f"Test Acc: {test_acc:.2f}%\n")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_checkpoint(model, "best_model.pth")
            print("✅ Best model saved!\n")

    print("🚀 Training Completed Successfully!")


if __name__ == "__main__":
    train()
