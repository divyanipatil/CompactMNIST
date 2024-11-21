import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class CompactCNN(nn.Module):
    def __init__(self):
        super(CompactCNN, self).__init__()
        # Input dim 28*28*1
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),  # weights 9*4=36 +4
            nn.BatchNorm2d(4, eps=1e-05,
    momentum=0.9,  # Increased from default 0.1
    affine=True),      # 8
            nn.ReLU(),
            nn.MaxPool2d(2),  # dim 7*7*8

            nn.Conv2d(4, 16, kernel_size=3, padding=1),  # weights 32*9=288 +8
            nn.BatchNorm2d(16, eps=1e-05,
    momentum=0.9,  # Increased from default 0.1
    affine=True),      # 16
            nn.ReLU(),
            nn.MaxPool2d(2),        # dim 7*7*8

            # nn.Conv2d(8, 12, kernel_size=3, padding=1),  # weights 96*9=864 +12
            # nn.BatchNorm2d(12),     # 24
            # nn.ReLU(),              # dim 7*7*12
            # # No pooling, cz dim is already too small 7*7
            #
            # nn.Conv2d(12, 16, kernel_size=3, padding=1),    # weights 192*9=1728 +16
            # nn.BatchNorm2d(16), # 32
            # nn.ReLU(),              # dim 7*7*16
            # No pooling, cz dim is already too small 7*7

            # nn.Conv2d(16, 20, kernel_size=3, padding=1),  # 2,900 params
            # nn.BatchNorm2d(20),  # 40 params
            # nn.ReLU(),
            # nn.MaxPool2d(2),  # dim 7*7*8
        )

        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 16, 30),  # weights (7*7*16*24)=18816 +24     # dim 24
            nn.ReLU(),
            nn.Linear(30, 10)           # weights (24*10)=240 +10           # dim 10
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CompactFullyConnectedNet(nn.Module):
    def __init__(self):
        super(CompactFullyConnectedNet, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # 28x28 -> 784

            nn.Linear(784, 32),  # 784*32 + 32 = 25,120
            nn.BatchNorm1d(32),  # 64
            nn.ReLU(),

            nn.Linear(32, 24),  # 32*24 + 24 = 792
            nn.BatchNorm1d(24),  # 48
            nn.ReLU(),

            nn.Linear(24, 10)  # 24*10 + 10 = 250
        )

    def forward(self, x):
        return self.fc_layers(x)


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
        batch_size=100,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Initialize model
    model =CompactCNN().to(device)

    # Calculate and print parameter count
    total_params = count_parameters(model)
    print("\nParameter count breakdown:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()}")
    print(f"\nTotal trainable parameters: {total_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.002, betas=(0.9, 0.999), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.02,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        epochs=1,
        div_factor=1.2,
        final_div_factor=10,
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