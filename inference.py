"""
Enhanced inference script for defect detection with crop-predict-merge approach.
This script:
1. Crops input image into 4 overlapping regions
2. Runs inference on each crop separately  
3. Merges predictions back to full image size
4. Handles overlapping regions using averaging
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
from typing import List, Tuple, Dict

from model.model import Unet


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_model(model_path, device, in_channels=3, num_classes=2):
    """Load the trained model from checkpoint."""
    model = Unet(in_channels=in_channels, num_classes=num_classes)
    
    # Load the state dict
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model


def calculate_crop_coordinates(img_shape: Tuple[int, int], crop_size: int = 256) -> List[Tuple[int, int, int, int]]:
    """
    Calculate coordinates for a grid of non-overlapping crops that cover the entire image.
    For 1024x1024 image with 256x256 crops, this creates a 4x4 grid (16 crops).
    
    Args:
        img_shape: (height, width) of the original image
        crop_size: Size of each square crop
    
    Returns:
        List of crop coordinates (x1, y1, x2, y2)
    """
    h, w = img_shape
    crops = []
    
    # Calculate number of crops in each dimension
    num_crops_h = h // crop_size
    num_crops_w = w // crop_size
    
    # Generate grid of crops
    for row in range(num_crops_h):
        for col in range(num_crops_w):
            x1 = col * crop_size
            y1 = row * crop_size
            x2 = x1 + crop_size
            y2 = y1 + crop_size
            
            # Ensure we don't exceed image boundaries
            x2 = min(x2, w)
            y2 = min(y2, h)
            
            crops.append((x1, y1, x2, y2))
    
    return crops


def crop_image(image: np.ndarray, crop_coords: Tuple[int, int, int, int], target_size: int = 256) -> np.ndarray:
    """
    Crop image and resize to target size if necessary.
    
    Args:
        image: Input image as numpy array (H, W, C)
        crop_coords: (x1, y1, x2, y2) crop coordinates
        target_size: Target size for the crop (will resize to this if crop is smaller)
    
    Returns:
        Cropped and potentially resized image
    """
    x1, y1, x2, y2 = crop_coords
    cropped = image[y1:y2, x1:x2]
    
    # For 256x256 crops from 1024x1024 image, no resizing should be needed
    # But handle edge cases where crop might be smaller due to image boundaries
    if cropped.shape[0] != target_size or cropped.shape[1] != target_size:
        cropped = cv2.resize(cropped, (target_size, target_size))
    
    return cropped


def preprocess_crop(crop: np.ndarray, cfg: Dict) -> torch.Tensor:
    """
    Preprocess a single crop for inference.
    
    Args:
        crop: Cropped image as numpy array (H, W, C)
        cfg: Configuration dictionary
    
    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, C, H, W)
    """
    # Convert to PIL Image
    image = PILImage.fromarray(crop).convert('RGB')
    
    # Get preprocessing parameters from config
    img_size = cfg['data']['img_size']
    mean = cfg['data']['mean']
    std = cfg['data']['std']
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((img_size, img_size)),
        transforms.GaussianBlur(kernel_size=(1, 1)),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    # Apply transforms
    image = Image(image)
    image = transform(image)
    
    # Add batch dimension
    image = image.unsqueeze(0)  # Shape: (1, C, H, W)
    
    return image


def postprocess_prediction(prediction: torch.Tensor, original_size: Tuple[int, int], threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Postprocess model prediction to create binary mask and confidence map.
    
    Args:
        prediction: Raw model output tensor of shape (1, num_classes, H, W)
        original_size: (height, width) of the original crop before resizing
        threshold: Threshold for binary classification
    
    Returns:
        Tuple of (binary_mask, confidence_map) as numpy arrays
    """
    # Apply softmax to get probabilities
    prediction = torch.softmax(prediction, dim=1)
    
    # Get defect probability (class 1)
    confidence_map = prediction[:, 1].squeeze().cpu().numpy()
    
    # Resize back to original crop size
    if original_size != confidence_map.shape:
        confidence_map = cv2.resize(confidence_map, (original_size[1], original_size[0]))
    
    # Apply threshold to create binary mask
    binary_mask = (confidence_map > threshold).astype(np.uint8) * 255
    
    return binary_mask, confidence_map


def merge_predictions(predictions: List[Dict], crop_coords: List[Tuple[int, int, int, int]], 
                     original_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge predictions from multiple crops back to original image size.
    For non-overlapping grid crops, this is straightforward placement.
    
    Args:
        predictions: List of prediction dictionaries from each crop
        crop_coords: List of crop coordinates
        original_shape: (height, width) of the original image
    
    Returns:
        Tuple of (merged_binary_mask, merged_confidence_map)
    """
    h, w = original_shape
    
    # Initialize output arrays
    merged_confidence = np.zeros((h, w), dtype=np.float32)
    merged_binary = np.zeros((h, w), dtype=np.uint8)
    
    # Place each prediction back to its original location
    for pred, (x1, y1, x2, y2) in zip(predictions, crop_coords):
        confidence_map = pred['confidence_map']
        binary_mask = pred['binary_mask']
        
        # Calculate actual crop dimensions
        crop_h, crop_w = y2 - y1, x2 - x1
        
        # Resize if necessary (for edge crops that might be smaller)
        if confidence_map.shape != (crop_h, crop_w):
            confidence_map = cv2.resize(confidence_map, (crop_w, crop_h))
            binary_mask = cv2.resize(binary_mask, (crop_w, crop_h))
        
        # Place back in original position
        merged_confidence[y1:y2, x1:x2] = confidence_map
        merged_binary[y1:y2, x1:x2] = binary_mask
    
    return merged_binary, merged_confidence


def run_crop_merge_inference(image_path: str, config_path: str, model_path: str = None, 
                            threshold: float = 0.5, output_dir: str = None, 
                            device: torch.device = None, crop_size: int = 256) -> Dict:
    """
    Run inference using crop-predict-merge approach with grid-based cropping.
    
    Args:
        image_path: Path to the input image
        config_path: Path to the configuration file
        model_path: Path to the trained model
        threshold: Threshold for binary classification
        output_dir: Directory to save results
        device: Device to run inference on
        crop_size: Size of each crop (256 for 1024x1024 -> 4x4 grid)
    
    Returns:
        Dictionary containing prediction results
    """
    # Load configuration
    cfg = load_config(config_path)
    
    # Set device
    if device is None:
        device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    print(f"Using device: {device}")
    print(f"Processing with grid-based crop-merge approach: {image_path}")
    print(f"Crop size: {crop_size}x{crop_size}")
    
    # Load model
    model = load_model(model_path, device, num_classes=cfg['data']['num_classes'])
    
    # Load and prepare original image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_shape = original_image.shape[:2]  # (height, width)
    
    print(f"Original image shape: {original_shape}")
    
    # Calculate crop coordinates
    crop_coords = calculate_crop_coordinates(original_shape, crop_size)
    
    # Process each crop
    crop_predictions = []
    
    for i, coords in enumerate(crop_coords):
        print(f"Processing crop {i+1}/{len(crop_coords)}: {coords}")
        
        # Crop image
        cropped = crop_image(original_image, coords, crop_size)
        
        # Preprocess crop
        input_tensor = preprocess_crop(cropped, cfg).to(device)
        
        # Run inference
        with torch.no_grad():
            prediction = model(input_tensor)
        
        # Postprocess prediction
        original_crop_size = (coords[3] - coords[1], coords[2] - coords[0])  # (h, w)
        binary_mask, confidence_map = postprocess_prediction(prediction, original_crop_size, threshold)
        
        crop_predictions.append({
            'binary_mask': binary_mask,
            'confidence_map': confidence_map,
            'coords': coords
        })
    
    # Merge predictions
    print("Merging predictions...")
    merged_binary, merged_confidence = merge_predictions(crop_predictions, crop_coords, original_shape)
    
    # Check if defects are detected
    has_defects = np.sum(merged_binary > 0) > 0
    
    # Print results
    print("\n" + "="*60)
    print("GRID-BASED CROP-MERGE INFERENCE RESULTS")
    print("="*60)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Original size: {original_shape[1]}x{original_shape[0]}")
    print(f"Crop size: {crop_size}x{crop_size}")
    print(f"Grid layout: {original_shape[0]//crop_size}x{original_shape[1]//crop_size}")
    print(f"Total crops processed: {len(crop_coords)}")
    print(f"Defects detected: {'Yes' if has_defects else 'No'}")
    print("="*60)
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save merged binary mask
        mask_path = os.path.join(output_dir, f"{image_name}_merged_mask.png")
        cv2.imwrite(mask_path, merged_binary)
        
        # Save merged confidence map
        confidence_path = os.path.join(output_dir, f"{image_name}_merged_confidence.png")
        confidence_uint8 = (merged_confidence * 255).astype(np.uint8)
        cv2.imwrite(confidence_path, confidence_uint8)
        
        # Create and save overlay image
        overlay = original_image.copy()
        overlay[merged_binary > 0] = [255, 0, 0]  # Red for defects
        blended = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)
        overlay_path = os.path.join(output_dir, f"{image_name}_merged_overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        
        # Save individual crop results for debugging (organized in grid layout)
        crops_dir = os.path.join(output_dir, f"{image_name}_crops")
        os.makedirs(crops_dir, exist_ok=True)
        
        # Calculate grid dimensions for organized saving
        grid_h = original_shape[0] // crop_size
        grid_w = original_shape[1] // crop_size
        
        crop_idx = 0
        for row in range(grid_h):
            for col in range(grid_w):
                if crop_idx < len(crop_predictions):
                    pred = crop_predictions[crop_idx]
                    coords = crop_coords[crop_idx]
                    
                    crop_name = f"crop_r{row:02d}_c{col:02d}"
                    
                    # Save crop image
                    x1, y1, x2, y2 = coords
                    crop_img = original_image[y1:y2, x1:x2]
                    crop_img_path = os.path.join(crops_dir, f"{crop_name}_image.png")
                    cv2.imwrite(crop_img_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
                    
                    # Save crop mask
                    crop_mask_path = os.path.join(crops_dir, f"{crop_name}_mask.png")
                    cv2.imwrite(crop_mask_path, pred['binary_mask'])
                    
                    # Save crop confidence
                    crop_conf_path = os.path.join(crops_dir, f"{crop_name}_confidence.png")
                    conf_uint8 = (pred['confidence_map'] * 255).astype(np.uint8)
                    cv2.imwrite(crop_conf_path, conf_uint8)
                    
                    crop_idx += 1
        
        # Handle any remaining crops (for edge cases)
        while crop_idx < len(crop_predictions):
            pred = crop_predictions[crop_idx]
            coords = crop_coords[crop_idx]
            
            crop_name = f"crop_edge_{crop_idx:02d}"
            
            # Save crop image
            x1, y1, x2, y2 = coords
            crop_img = original_image[y1:y2, x1:x2]
            crop_img_path = os.path.join(crops_dir, f"{crop_name}_image.png")
            cv2.imwrite(crop_img_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
            
            # Save crop mask
            crop_mask_path = os.path.join(crops_dir, f"{crop_name}_mask.png")
            cv2.imwrite(crop_mask_path, pred['binary_mask'])
            
            # Save crop confidence
            crop_conf_path = os.path.join(crops_dir, f"{crop_name}_confidence.png")
            conf_uint8 = (pred['confidence_map'] * 255).astype(np.uint8)
            cv2.imwrite(crop_conf_path, conf_uint8)
            
            crop_idx += 1
        
        print(f"\nResults saved to: {output_dir}")
        print(f"  - Merged mask: {mask_path}")
        print(f"  - Merged confidence: {confidence_path}")
        print(f"  - Merged overlay: {overlay_path}")
        print(f"  - Individual crops: {crops_dir}")
    
    # Prepare return dictionary
    results = {
        'merged_binary_mask': merged_binary,
        'merged_confidence_map': merged_confidence,
        'crop_predictions': crop_predictions,
        'crop_coordinates': crop_coords,
        'has_defects': has_defects,
        'defect_percentage': defect_percentage,
        'original_shape': original_shape,
        'model_path': model_path,
        'config': cfg
    }
    
    return results


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='Run defect detection inference with crop-merge approach')
    parser.add_argument('--image_path', type=str, help='Path to the input image',
                        default='bracket_black/test/hole/000.png')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--model', type=str, 
                       default='weights/experiment_22_bs10_lr0.0009_wd_1e-5_baseline_cropped_images_IOULoss_fixed_IOU/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary classification (default: 0.5)')
    parser.add_argument('--output_dir', type=str, default='./output_crop_merge',
                       help='Directory to save results')
    parser.add_argument('--crop_size', type=int, default=256,
                       help='Size of crops for processing (default: 256 for 1024x1024 images)')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run crop-merge inference
    results = run_crop_merge_inference(
        image_path=args.image_path, 
        config_path=args.config,
        model_path=args.model, 
        threshold=args.threshold,
        output_dir=args.output_dir, 
        device=device,
        crop_size=args.crop_size
    )
    
    return results


if __name__ == "__main__":
    main()
