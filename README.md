# Part Defect Detection

A deep learning project for detecting defects in manufacturing parts using semantic segmentation. This project implements a U-Net based model to identify and segment defects such as holes and scratches in bracket parts.

## Overview

This repository contains a complete pipeline for training and evaluating a defect detection model for manufacturing quality control. The model is trained to perform binary segmentation to identify defective regions in part images.

### Supported Defect Types
- **Holes**: Missing material or punctures in parts
- **Scratches**: Surface damage and scoring
- **Good**: Non-defective reference samples

## Project Structure

```
├── config.yaml              # Configuration file for training parameters
├── main.py                  # Main training script
├── train.py                 # Training and validation functions
├── dataloader.py            # Custom dataset class and data loading
├── loss.py                  # Custom loss functions
├── inference.py             # Model inference script
├── preprocess.py            # Sliding window preprocessing for defect-centric patches
├── model/
│   ├── model.py             # U-Net model implementation
│   └── utils.py             # Model utility functions
├── tests/                   # Unit tests
├── bracket_black/           # Dataset directory
│   ├── train/               # Training images
│   ├── test/                # Test images
│   └── ground_truth/        # Ground truth masks
└── weights/                 # Saved model weights
```

## Features

- **U-Net Architecture**: Deep convolutional neural network optimized for semantic segmentation
- **Mixed Precision Training**: Supports both bfloat16 and float16 for faster training
- **Multiple Loss Functions**: Currently implements Dice Loss for segmentation
- **WandB Integration**: Experiment tracking and logging
- **Data Augmentation**: Configurable augmentation pipeline
- **Automated Model Saving**: Saves best models based on validation loss

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Configuration

Edit the `config.yaml` file to set your training parameters:

```yaml
data:
  num_classes: 1          # Binary segmentation
  img_size: 640           # Input image size
  mean: [0.436, 0.637, 0.111]  # Dataset normalization values
  std: [0.140, 0.182, 0.0902]

train:
  train_dir: bracket_black/test    # Path to training data
  batch_size: 10
  epochs: 150
  learning_rate: 0.005
  weight_decay: 1e-5
  test_train_split: 0.95          # 95% train, 5% validation
```

### 2. Training

Run the training script:

```bash
python main.py
```

The training script will:
- Load and split the dataset into train/validation sets
- Initialize the U-Net model
- Train the model with mixed precision
- Save the best model weights based on validation loss
- Log metrics to WandB (if enabled)

### 3. Model Architecture

The model uses a U-Net architecture with:
- **Encoder**: 4 downsampling blocks (64→128→256→512 channels)
- **Bottleneck**: 1024 channels
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: Single channel for binary segmentation

### 4. Loss Function

The model uses Dice Loss, which is particularly effective for segmentation tasks with class imbalance:

```python
dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)
```

### 5. Inference

For inference on new images, you can use the inference script with various parameters:

```bash
python inference.py --image_path <path_to_image> [OPTIONS]
```

#### Available Parameters:

- `--image_path`: Path to the input image for inference
  - Default: `bracket_black/test/hole/000.png`
  - Example: `--image_path path/to/your/image.jpg`

- `--config`: Path to the configuration YAML file
  - Default: `config.yaml`
  - Example: `--config custom_config.yaml`

- `--model`: Path to the trained model weights
  - Default: `weights/experiment_9_bs10_lr0.005_wd_1e-5_baseline_without_jitter_blurInVal_kernel5/best_model.pth`
  - Example: `--model weights/my_model.pth`

- `--threshold`: Threshold for binary classification (0.0 to 1.0)
  - Default: `0.5`
  - Example: `--threshold 0.3` (more sensitive to defects)

- `--output_dir`: Directory to save inference results
  - Default: `./output`
  - Example: `--output_dir results/inference_run_1`

#### Example Usage:

```bash
# Basic inference with default parameters
python inference.py

# Inference on a specific image with custom threshold
python inference.py --image_path my_test_image.jpg --threshold 0.3

# Full parameter specification
python inference.py \
    --image_path bracket_black/test/scratches/005.png \
    --config config.yaml \
    --model weights/my_best_model.pth \
    --threshold 0.4 \
    --output_dir results/scratches_test
```

#### Output Files:

The inference script generates the following output files in the specified output directory:

- `{image_name}_prediction_mask.png`: Binary prediction mask (white = defect, black = good)
- `{image_name}_confidence_map.png`: Confidence/probability map (grayscale)
- `{image_name}_overlay.png`: Original image with defects highlighted in red

#### Programmatic Usage:

You can also use the inference function programmatically:

```python
from inference import run_inference

results = run_inference(
    image_path='path/to/image.jpg',
    config_path='config.yaml',
    model_path='weights/best_model.pth',
    threshold=0.5,
    output_dir='./results'
)

# Access results
has_defects = results['has_defects']
prediction_mask = results['prediction_mask']
confidence_map = results['confidence_map']
```

## Data Format

The dataset should be organized as follows:
```
bracket_black/
├── train/
│   ├── good/           # Non-defective samples
│   ├── hole/           # Samples with hole defects
│   └── scratches/      # Samples with scratch defects
├── test/
│   ├── good/
│   ├── hole/
│   └── scratches/
└── ground_truth/       # Segmentation masks
    ├── hole/
    └── scratches/
```

## Experiment Tracking

The project supports WandB for experiment tracking. Enable it in `config.yaml`:pre

```yaml
wandb:
  use_wandb: True
  project: your_project_name
```

## Model Weights

Trained model weights are automatically saved in the `weights/` directory with experiment details:
- Best model based on validation loss
- Experiment configuration
- Training code snapshot

## Testing

Run the test suite:

```bash
pytest tests/
```

## Future Work

### Sliding Window Preprocessing Enhancement

A promising preprocessing methodology to improve model training through defect-centric patch extraction:

#### Methodology Overview
- **Defect-Focused Cropping**: Extract 480×480 patches centered around defective regions instead of training on full images
- **Multi-Sample Generation**: Generate 4 crops per defective image with positional offsets to increase training data
- **Balanced Sampling**: Create 3 random crops from good images to maintain class balance
- **Resolution Optimization**: Use fixed-size patches for consistent input dimensions and better computational efficiency

#### Expected Benefits
- **Enhanced Feature Learning**: Higher pixel density for defect regions improves small defect detection
- **Natural Data Augmentation**: 4x increase in defective samples through systematic spatial variance
- **Computational Efficiency**: Smaller input tensors enable larger batch sizes and faster training
- **Class Balance**: Addresses inherent imbalance between defective and good samples

This approach transforms the training paradigm from global scene understanding to focused patch-based learning, potentially improving detection accuracy for small manufacturing defects.

## License

This project is licensed under the terms specified in the LICENSE file.