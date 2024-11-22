# MNIST Classification with PyTorch
![Build pass](https://github.com/divyanipatil/CompactMNIST/actions/workflows/model_validation.yml/badge.svg)

A deep learning project implementing CNN for MNIST digit classification, optimized for parameter efficiency while maintaining high accuracy.

## 🎯 Project Goals

- Implement a CNN model with less than 25,000 parameters
- Achieve classification accuracy of ≥95% on MNIST dataset

## 🛠️ Technologies Used

- Python 3.8
- PyTorch
- torchvision
- pytest for testing

## 📦 Installation

1. Clone the repository:
```bash
    git clone https://github.com/divyanipatil/CompactMNIST.git
```

2. Install dependencies:
```bash
    pip install -r requirements.txt
```
## 🏗️ Project Structure

- `train.py` - CNN implementation and training logic
- `test_model.py` - Model testing and validation
- GitHub Actions workflow for automated testing

## 🚀 Models

### CompactCNN
- Convolutional Neural Network optimized for parameter efficiency
- Features batch normalization and ReLU activation
- Uses Adam optimizer with OneCycleLR scheduler

## 💻 Usage

To train the CNN model:
```bash
    python train.py
```

To run tests:
```bash
    pytest test_model.py -v
```

## 🔍 Model Performance

The model is designed to meet the following criteria:
- Parameter count: < 25,000
- Accuracy threshold: ≥95%

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚙️ CI/CD

The project includes GitHub Actions workflow for automated testing:
- Runs on Ubuntu latest
- Uses Python 3.8
- Automatically runs tests on push to main and pull requests
- Validates two critical aspects:
     - Model parameter count (must be < 25,000)
     - Model accuracy (must be ≥ 95%)