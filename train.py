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
        self.features = nn.Sequential(
            # First conv block: 28x28x1 -> 28x28x6
            nn.Conv2d(1, 4, kernel_size=3, padding=1),  # (3*3*1*6) + 6 = 60
            nn.BatchNorm2d(4),  # 12
            nn.ReLU(),  # 0
            nn.MaxPool2d(2),  # 0

            # Second conv block: 14x14x6 -> 14x14x12
            nn.Conv2d(4, 8, kernel_size=3, padding=1),  # (3*3*6*12) + 12 = 660
            nn.BatchNorm2d(8),  # 24
            nn.ReLU(),  # 0
            nn.MaxPool2d(2),  # 0

            # Third conv block: 7x7x12 -> 7x7x16
            nn.Conv2d(8, 12, kernel_size=3, padding=1),  # (3*3*12*16) + 16 = 1,744
            nn.BatchNorm2d(12),  # 32
            nn.ReLU(),  # 0

            # Fourth conv block: 14x14x12 -> 14x14x16
            nn.Conv2d(12, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 16, 32),  # (7*7*16*32) + 32 = 25,120
            nn.ReLU(),  # 0
            nn.Linear(32, 10)  # (32*10) + 10 = 330
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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.05,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        epochs=1,
        anneal_strategy='cos'
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

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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