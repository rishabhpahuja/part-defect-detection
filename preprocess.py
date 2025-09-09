"""
Preprocessing script to crop images around defective areas.
This script extracts 480x480 patches centered on defective regions
to provide a zoomed-in view of defects for better training.
"""

import os
import numpy as np
from PIL import Image
import cv2
import argparse


def find_defect_bbox(mask_path, min_area=5):
    """
    Find bounding box around defective pixels in the mask.
    
    Args:
        mask_path: Path to the ground truth mask
        min_area: Minimum area of defect to consider
    
    Returns:
        tuple: (x, y, w, h) bounding box or None if no significant defect found
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    # Find contours of defective areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Filter contours by area and merge them
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    if not valid_contours:
        return None
    
    # Get bounding box that encompasses all valid contours
    all_points = np.vstack(valid_contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    return (x, y, w, h)


def calculate_crop_regions(bbox, img_shape, crop_size=480, num_crops=4):
    """
    Calculate multiple crop regions around the defect bounding box.
    
    Args:
        bbox: (x, y, w, h) bounding box of defect
        img_shape: (height, width) of the original image
        crop_size: Size of the square crop
        num_crops: Number of crops to generate around the defect
    
    Returns:
        list: List of (x1, y1, x2, y2) crop coordinates
    """
    x, y, w, h = bbox
    img_h, img_w = img_shape
    
    # Center of the defect
    center_x = x + w // 2
    center_y = y + h // 2
    
    crop_regions = []
    
    # Define offset patterns for multiple crops around the defect
    # Center crop, and then offset crops in different directions
    offsets = [
        (0, 0),                    # Center crop
        (-crop_size//4, -crop_size//4),  # Top-left offset
        (crop_size//4, -crop_size//4),   # Top-right offset
        (0, crop_size//4),               # Bottom offset
    ]
    
    for i, (offset_x, offset_y) in enumerate(offsets[:num_crops]):
        # Apply offset to center
        crop_center_x = center_x + offset_x
        crop_center_y = center_y + offset_y
        
        # Calculate crop boundaries
        half_crop = crop_size // 2
        x1 = max(0, crop_center_x - half_crop)
        y1 = max(0, crop_center_y - half_crop)
        x2 = min(img_w, crop_center_x + half_crop)
        y2 = min(img_h, crop_center_y + half_crop)
        
        # Adjust if crop goes outside image boundaries
        if x2 - x1 < crop_size:
            if x1 == 0:
                x2 = min(img_w, crop_size)
            else:
                x1 = max(0, img_w - crop_size)
                x2 = img_w
        
        if y2 - y1 < crop_size:
            if y1 == 0:
                y2 = min(img_h, crop_size)
            else:
                y1 = max(0, img_h - crop_size)
                y2 = img_h
        
        crop_regions.append((x1, y1, x2, y2))
    
    return crop_regions


def crop_image(image_path, crop_coords, output_path):
    """
    Crop image according to the specified coordinates.
    
    Args:
        image_path: Path to the original image
        crop_coords: (x1, y1, x2, y2) crop coordinates
        output_path: Path to save the cropped image
    """
    image = Image.open(image_path)
    x1, y1, x2, y2 = crop_coords
    cropped = image.crop((x1, y1, x2, y2))
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cropped.save(output_path)


def process_defective_images(data_root, output_root, defect_type, split, crop_size=480, num_crops=4):
    """
    Process images with defects by cropping around defective areas.
    Also crops and saves the corresponding masks.
    
    Args:
        data_root: Root directory of the dataset
        output_root: Root directory for processed dataset
        defect_type: Type of defect ('hole' or 'scratches')
        split: Dataset split ('train' or 'test')
        crop_size: Size of the square crop
        num_crops: Number of crops to generate around each defect
    """
    image_dir = os.path.join(data_root, split, defect_type)
    mask_dir = os.path.join(data_root, 'ground_truth', defect_type)
    output_image_dir = os.path.join(output_root, split, defect_type)
    output_mask_dir = os.path.join(output_root, 'ground_truth', defect_type)
    
    if not os.path.exists(image_dir):
        print(f"Image directory not found: {image_dir}")
        return
    
    if not os.path.exists(mask_dir):
        print(f"Mask directory not found: {mask_dir}")
        return
    
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    processed_count = 0
    skipped_count = 0
    
    # Process each image
    for image_file in sorted(os.listdir(image_dir)):
        if not image_file.endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # Find corresponding mask
        base_name = os.path.splitext(image_file)[0]
        mask_file = f"{base_name}_mask.png"
        
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, mask_file)
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {image_file}, skipping...")
            skipped_count += 1
            continue
        
        # Find defect bounding box
        bbox = find_defect_bbox(mask_path)
        if bbox is None:
            print(f"Warning: No significant defect found in {mask_file}, skipping...")
            skipped_count += 1
            continue
        
        # Load image to get dimensions
        img = Image.open(image_path)
        img_shape = (img.height, img.width)
        
        # Calculate multiple crop regions around the defect
        crop_regions = calculate_crop_regions(bbox, img_shape, crop_size, num_crops)
        
        # Crop and save multiple images and their corresponding masks
        for i, crop_coords in enumerate(crop_regions):
            # Crop and save image
            output_image_filename = f"{base_name}_crop_{i:02d}.png"
            output_image_path = os.path.join(output_image_dir, output_image_filename)
            crop_image(image_path, crop_coords, output_image_path)
            
            # Crop and save corresponding mask
            output_mask_filename = f"{base_name}_crop_{i:02d}_mask.png"
            output_mask_path = os.path.join(output_mask_dir, output_mask_filename)
            crop_image(mask_path, crop_coords, output_mask_path)
            
            processed_count += 1
        
        print(f"Processed: {image_file} -> {num_crops} crops (images + masks)")
    
    print(f"Completed {defect_type} {split}: {processed_count} total crops, {skipped_count} skipped")


def process_good_images(data_root, output_root, split, crop_size=480, num_crops_per_image=3):
    """
    Process good images by taking random crops.
    
    Args:
        data_root: Root directory of the dataset
        output_root: Root directory for processed dataset
        split: Dataset split ('train' or 'test')
        crop_size: Size of the square crop
        num_crops_per_image: Number of random crops per good image
    """
    image_dir = os.path.join(data_root, split, 'good')
    output_dir = os.path.join(output_root, split, 'good')
    
    if not os.path.exists(image_dir):
        print(f"Good image directory not found: {image_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    
    # Process each good image
    for image_file in sorted(os.listdir(image_dir)):
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        image_path = os.path.join(image_dir, image_file)
        img = Image.open(image_path)
        img_w, img_h = img.size
        
        # Skip if image is smaller than crop size
        if img_w < crop_size or img_h < crop_size:
            print(f"Warning: Image {image_file} is too small for cropping, skipping...")
            continue
        
        # Generate random crops
        base_name = os.path.splitext(image_file)[0]
        for i in range(num_crops_per_image):
            # Random crop coordinates
            x1 = np.random.randint(0, img_w - crop_size + 1)
            y1 = np.random.randint(0, img_h - crop_size + 1)
            x2 = x1 + crop_size
            y2 = y1 + crop_size
            
            crop_coords = (x1, y1, x2, y2)
            output_filename = f"{base_name}_crop_{i:02d}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            crop_image(image_path, crop_coords, output_path)
            processed_count += 1
        
        print(f"Processed: {image_file} -> {num_crops_per_image} crops")
    
    print(f"Completed good {split}: {processed_count} total crops")

def reduce_glare_using_impaint(img_path:str, mask_seg_path:str = None):
    """
    Reduce glare in the image using inpainting and saves in place.
    Args:
        img_path: Path to the directory containing images
        mask_seg_path: Path to the directory containing segmentation masks (optional)
    """

    for image in os.listdir(img_path):

        img = cv2.imread(os.path.join(img_path, image))

        # convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # threshold grayscale image to extract glare
        mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        if mask_seg_path is not None:
            mask_path = os.path.join(mask_seg_path, image.split('.')[0] + '_mask.png')
            mask_seg = cv2.imread(mask_path, 0) if os.path.exists(mask_path) else None # Open in grayscale
            mask[mask_seg == 255] = 0

        # use mask with input to do inpainting
        result = cv2.inpaint(img, mask, 21, cv2.INPAINT_NS) 
        cv2.imwrite(os.path.join(img_path, image), result)


def main():
    parser = argparse.ArgumentParser(description='Preprocess defect detection dataset')
    parser.add_argument('--data_root', type=str, default='bracket_black_processed',
                        help='Root directory of the original dataset')
    parser.add_argument('--output_root', type=str, default='bracket_black_cropped',
                        help='Root directory for the processed dataset')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='Size of the square crop')
    parser.add_argument('--splits', nargs='+', default=[ 'test'],
                        help='Dataset splits to process')
    parser.add_argument('--defect_types', nargs='+', default=['hole', 'scratches', 'good'],
                        help='Types of defects to process')
    parser.add_argument('--defect_crops_per_image', type=int, default=4,
                        help='Number of crops around each defect')
    parser.add_argument('--good_crops_per_image', type=int, default=3,
                        help='Number of random crops per good image')
    parser.add_argument('--remove_flash,', action='store_true',
                        help='Remove flash images if present')
    
    args = parser.parse_args()
    
    print(f"Starting preprocessing...")
    print(f"Input: {args.data_root}")
    print(f"Output: {args.output_root}")
    print(f"Crop size: {args.crop_size}x{args.crop_size}")
    print(f"Splits: {args.splits}")
    print(f"Defect types: {args.defect_types}")
    print(f"Defect crops per image: {args.defect_crops_per_image}")
    print(f"Good crops per image: {args.good_crops_per_image}")
    print("-" * 50)
    
    # Process each split and defect type
    for split in args.splits:
        print(f"\nProcessing {split} split...")
        
        # Process defective images
        for defect_type in args.defect_types:
            
            print(f"Reducing glare in {defect_type} defects...")
            reduce_glare_using_impaint(
                img_path = os.path.join(args.data_root, split, defect_type),
            )
            print(f"Processing {defect_type} defects...")
            process_defective_images(
                args.data_root, args.output_root, defect_type, split, 
                args.crop_size, args.defect_crops_per_image
            )
        
        # Process good images
        print(f"Processing good images...")
        process_good_images(
            args.data_root, args.output_root, split, args.crop_size, args.good_crops_per_image
        )
    
    print(f"\nPreprocessing completed! Processed dataset saved to: {args.output_root}")


if __name__ == "__main__":
    main()
