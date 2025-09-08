#!/usr/bin/env python3
"""
Inference script for defect detection baseline model.
Runs inference on a single image and generates prediction mask.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
import yaml
import os
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from torchvision.transforms import v2 as transforms
from torchvision.tv_tensors import Image

from model.model import Unet


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_model(model_path, device, in_channels=3, num_classes=1):
    """
    Load the trained model from checkpoint.
    
    """
    model = Unet(in_channels=in_channels, num_classes=num_classes)
    
    # Load the state dict
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")

    return model


def preprocess_image(image_path, cfg):
    """
    Preprocess a single image for inference.
    
    Args:
        image_path: Path to the input image
        cfg: Configuration dictionary
    
    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, C, H, W)
    """
    # Load image
    image = PILImage.open(image_path).convert('RGB')
    
    # Get preprocessing parameters from config
    img_size = cfg['data']['img_size']
    mean = cfg['data']['mean']
    std = cfg['data']['std']
    
    # Define transforms (similar to validation transforms)
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((img_size, img_size)),
        transforms.GaussianBlur(kernel_size=(5, 5)),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    # Apply transforms
    image = Image(image)
    image = transform(image)
    
    # Add batch dimension
    image = image.unsqueeze(0)  # Shape: (1, C, H, W)
    
    return image


def postprocess_prediction(prediction, threshold=0.5):
    """
    Postprocess model prediction to create binary mask.
    
    Args:
        prediction: Raw model output tensor of shape (1, 1, H, W)
        threshold: Threshold for binary classification
    
    Returns:
        numpy.ndarray: Binary mask of shape (H, W)
    """
    # Apply sigmoid activation if not already applied
    prediction = torch.sigmoid(prediction)
    
    # Remove batch and channel dimensions
    prediction = prediction.squeeze().cpu().numpy()
    
    # Apply threshold to create binary mask
    binary_mask = (prediction > threshold).astype(np.uint8) * 255
    
    return binary_mask, prediction



def run_inference(image_path, config_path, model_path=None, threshold=0.5, 
                 output_dir=None, device=None):
    """
    Run inference on a single image.
    
    Args:
        image_path: Path to the input image
        config_path: Path to the configuration file
        model_path: Path to the trained model (optional, will use config if not provided)
        threshold: Threshold for binary classification
        output_dir: Directory to save results
        device: Device to run inference on
    
    Returns:
        dict: Dictionary containing prediction results and metrics
    """
    # Load configuration
    cfg = load_config(config_path)
    
    # Set device
    if device is None:
        device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    print(f"Using device: {device}")
        
    # Load model
    model = load_model(model_path, device, num_classes=cfg['data']['num_classes'])
    
    # Preprocess image
    print(f"Processing image: {image_path}")
    input_tensor = preprocess_image(image_path, cfg).to(device)
    
    # Run inference
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # Postprocess prediction
    binary_mask, confidence_map = postprocess_prediction(prediction, threshold)
    
    # Check if defects are detected
    has_defects = np.sum(binary_mask > 0) > 0
    
    # Print results
    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Defects detected: {'Yes' if has_defects else 'No'}")
    print("="*50)
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save binary mask
        mask_path = os.path.join(output_dir, f"{image_name}_prediction_mask.png")
        cv2.imwrite(mask_path, binary_mask)
        
        # Save confidence map
        confidence_path = os.path.join(output_dir, f"{image_name}_confidence_map.png")
        confidence_uint8 = (confidence_map * 255).astype(np.uint8)
        cv2.imwrite(confidence_path, confidence_uint8)
        
        # Create and save overlay image
        original_image = PILImage.open(image_path).convert('RGB')
        original_image = np.array(original_image)
        h, w = binary_mask.shape
        original_image_resized = cv2.resize(original_image, (w, h))
        overlay = original_image_resized.copy()
        overlay[binary_mask > 0] = [255, 0, 0]  # Red for defects
        blended = cv2.addWeighted(original_image_resized, 0.7, overlay, 0.3, 0)
        overlay_path = os.path.join(output_dir, f"{image_name}_overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        
        print(f"Results saved to: {output_dir}")
        print(f"  - Prediction mask: {mask_path}")
        print(f"  - Confidence map: {confidence_path}")
        print(f"  - Overlay image: {overlay_path}")

    
    # Prepare return dictionary
    results = {
        'prediction_mask': binary_mask,
        'confidence_map': confidence_map,
        'has_defects': has_defects,
        'model_path': model_path,
        'config': cfg
    }
    
    return results


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='Run defect detection inference on a single image')
    parser.add_argument('--image_path', type=str, help='Path to the input image',
                        default='bracket_black/test/hole/000.png')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--model', type=str, default='weights/experiment_9_bs10_lr0.005_wd_1e-5_baseline_without_jitter_blurInVal_kernel5/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary classification (default: 0.5)')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Directory to save results')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run inference
    results = run_inference(image_path=args.image_path, config_path=args.config,
                            model_path=args.model, threshold=args.threshold,
                            output_dir=args.output_dir, device=device
    )
    
    return results


if __name__ == "__main__":
    main()