import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        # Let's calculate parameters for each layer:
        self.features = nn.Sequential(
            # Conv1: (3x3x1x16) + 16 bias = 160 params
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            # BatchNorm1: 2x16 (mean and var) = 32 params
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv2: (3x3x16x32) + 32 bias = 4,640 params
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            # BatchNorm2: 2x32 = 64 params
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            # FC1: (7x7x32)x128 + 128 bias = 25,088 params - Too many!
            # Let's reduce to 64 nodes
            nn.Linear(7 * 7 * 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            # FC2: 64x10 + 10 bias = 650 params
            nn.Linear(64, 10)
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
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Initialize model
    model = ImprovedCNN().to(device)

    # Calculate and print parameter count
    total_params = count_parameters(model)
    print("\nParameter count breakdown:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()}")
    print(f"\nTotal trainable parameters: {total_params}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.003,
        steps_per_epoch=len(train_loader),
        epochs=1
    )

    # Training
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Batch: {batch_idx}/{len(train_loader)}, '
                  f'Loss: {running_loss / (batch_idx + 1):.4f}, '
                  f'Accuracy: {100. * correct / total:.2f}%, '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')

    final_accuracy = 100. * correct / total
    print(f'Final Training Accuracy: {final_accuracy:.2f}%')

    return total_params, final_accuracy


if __name__ == "__main__":
    params, accuracy = train_model()

    # Save results for GitHub Actions
    with open("results.txt", "w") as f:
        f.write(f"Parameters: {params}\nAccuracy: {accuracy}")