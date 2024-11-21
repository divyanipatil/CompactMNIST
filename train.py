import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class CompactCNN(nn.Module):
    def __init__(self):
        super(CompactCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 28x28x8
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14x8
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 14x14x16
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7x16
        )
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 16, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and transform data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True
    )

    # Initialize model
    model = CompactCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Print model parameters
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params}")

    # Training
    model.train()
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 0:
            print(f'Batch: {batch_idx}, Loss: {loss.item():.4f}, '
                  f'Accuracy: {100. * correct / total:.2f}%')

    final_accuracy = 100. * correct / total
    print(f'Final Training Accuracy: {final_accuracy:.2f}%')

    return total_params, final_accuracy


if __name__ == "__main__":
    params, accuracy = train_model()

    # Save results for GitHub Actions
    with open("results.txt", "w") as f:
        f.write(f"Parameters: {params}\nAccuracy: {accuracy}")