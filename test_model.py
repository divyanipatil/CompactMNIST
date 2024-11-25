import time
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from .train import CompactCNN, count_parameters, train_model

model_path = 'models/mnist_model.pth'
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def test_parameter_count():
    model = CompactCNN()
    param_count = count_parameters(model)
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"


def test_model_accuracy():
    params, accuracy = train_model()
    assert accuracy >= 95.0, f"Model accuracy is {accuracy}%, should be at least 95%"


def test_model_inference_speed():
    """Test if model inference time is within acceptable limits"""
    model = CompactCNN()
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.eval()

    # Prepare sample input
    sample_input = torch.randn(1, 1, 28, 28)  # Single MNIST image shape

    # Measure inference time for 100 forward passes
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(sample_input)
    avg_time = (time.time() - start_time) / 100

    assert avg_time < 0.005, f"Inference too slow: {avg_time:.4f}s per image (should be <0.005s)"


def test_model_robustness():
    """Test model's robustness to input noise"""
    model = CompactCNN()
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.eval()

    # Load a single test image
    test_loader = DataLoader(
        torchvision.datasets.MNIST('data', train=False, download=True, transform=test_transform),
        batch_size=1
    )
    image, label = next(iter(test_loader))

    # Test prediction with different noise levels
    noise_levels = [0.1, 0.2, 0.3]
    original_pred = model(image).argmax(dim=1)

    for noise_level in noise_levels:
        noisy_image = image + noise_level * torch.randn_like(image)
        noisy_pred = model(noisy_image).argmax(dim=1)

        # Model should maintain prediction for reasonable noise levels
        assert noisy_pred == original_pred, f"Model prediction changed with noise level {noise_level}"


def test_model_confidence():
    """Test if model's confidence aligns with accuracy"""
    model = CompactCNN()
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.eval()

    # Get a batch of test data
    test_loader = DataLoader(
        torchvision.datasets.MNIST('data', train=False, download=True, transform=test_transform),
        batch_size=100,
    )
    images, labels = next(iter(test_loader))

    with torch.no_grad():
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)

        # Get predictions and their confidences
        predictions = probs.argmax(dim=1)
        confidences = probs.max(dim=1).values

        # Check correct predictions
        correct = predictions == labels

        # Average confidence for correct and incorrect predictions
        correct_conf = confidences[correct].mean()
        if len(confidences[~correct]) > 0:
            incorrect_conf = confidences[~correct].mean()

            # Confidence should be higher for correct predictions
            assert correct_conf > incorrect_conf, \
                f"Model confidence for correct predictions ({correct_conf:.3f}) should be higher than incorrect ones ({incorrect_conf:.3f})"

            # Confidence gap should be significant
            assert (correct_conf - incorrect_conf) > 0.3, \
                f"Confidence gap ({correct_conf - incorrect_conf:.3f}) should be more than 0.3"