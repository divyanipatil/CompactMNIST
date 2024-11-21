# test_model.py
import pytest
from train import CompactCNN, count_parameters, train_model


def test_parameter_count():
    model = CompactCNN()
    param_count = count_parameters(model)
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"


def test_model_accuracy():
    params, accuracy = train_model()
    assert accuracy >= 95.0, f"Model accuracy is {accuracy}%, should be at least 95%"