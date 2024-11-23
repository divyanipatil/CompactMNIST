import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import random

# Load the MNIST dataset
dataset = MNIST(root='./data', train=True, download=True)

# Define the transformations
transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=10,
        translate=(0.1, 0.1),
        scale=(0.9, 1.0)
    ),
    transforms.Normalize((0.1307,), (0.3081,)),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor()
])

# Apply transformations to a subset of the dataset
# Randomly select 3 images from the dataset
indices = random.sample(range(len(dataset)), 5)
original_images = [transforms.ToTensor()(dataset[i][0]) for i in indices]
transformed_images = [transform(dataset[i][0]) for i in indices]
original_images = [transforms.ToTensor()(dataset[i][0]) for i in indices]


# Original images without transformations

# Plot the images
fig, axs = plt.subplots(2, 5, figsize=(12, 6))

for i in range(5):
    axs[0, i].imshow(original_images[i].squeeze(), cmap='gray')
    axs[0, i].axis('off')
    axs[0, i].set_title('Original')

    axs[1, i].imshow(transformed_images[i].squeeze(), cmap='gray')
    axs[1, i].axis('off')
    axs[1, i].set_title('Augmented')

# Save the plot
# plt.savefig('mnist_augmented_images6.png')
# plt.show()
