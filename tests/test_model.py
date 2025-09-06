import torch
import pytest
from model.model import Model

def test_model_output_shape():
    # Test parameters
    batch_size = 4
    in_channels = 3
    num_classes = 1
    height = 256
    width = 256

    # Create model instance
    model = Model(in_channels=in_channels, num_classes=num_classes)
    
    # Create dummy input tensor
    x = torch.randn(batch_size, in_channels, height, width)
    
    # Get model output
    output = model(x)
    
    # Check output shape
    expected_shape = (batch_size, num_classes, height, width)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"

def test_model_different_input_sizes():
    # Test with different input sizes
    test_sizes = [(1, 512, 512), (8, 128, 128)]
    in_channels = 3
    num_classes = 1
    
    model = Model(in_channels=in_channels, num_classes=num_classes)
    
    for batch_size, height, width in test_sizes:
        x = torch.randn(batch_size, in_channels, height, width)
        output = model(x)
        expected_shape = (batch_size, num_classes, height, width)
        assert output.shape == expected_shape, \
            f"Failed for input size {(batch_size, height, width)}: Expected shape {expected_shape}, got {output.shape}"