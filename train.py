import torch
from torch.utils.data import DataLoader
from loss import DefectSegmentationLoss
import wandb
import numpy as np
from PIL import Image as PILImage

def train(data_loader: DataLoader, model:torch.nn.Module, 
            criterion:DefectSegmentationLoss,
            optimizer:torch.optim.Optimizer, device:str,
            logger:wandb, cfg:dict)->float:

    '''
    Trains the model for one epoch
    Args:
        data_loader: DataLoader for training data
        model: The neural network model
        criterion: Loss function
        optimizer: Optimizer for updating model weights
        device: Device to run the training on (CPU or GPU)
    Returns:
        Average loss for the epoch
    '''

    model.train()
    running_loss = 0.0

    for images, masks in data_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(data_loader)
    
    if cfg['use_wandb']:
        logger.log({"train/loss": epoch_loss}, commit=False)

    return epoch_loss

def validate(data_loader: DataLoader, model:torch.nn.Module, logger:wandb,
            criterion:DefectSegmentationLoss, device:str, cfg: dict)->float:

    '''
    Validates the model for one epoch
    Args:
        data_loader: DataLoader for validation data
        model: The neural network model
        criterion: Loss function
        device: Device to run the validation on (CPU or GPU)
    Returns:
        Average loss for the epoch
    '''

    model.eval()
    running_loss = 0.0
    logged_batch = False

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item()

            if cfg['use_wandb']:
                # Log a few validation examples from the 1st batch to wandb
                if not logged_batch:
                    probs = torch.sigmoid(outputs)                 
                    preds = (probs > cfg['data']['class_threshold']).float()
                    B = images.size(0)
                    n = min(3, B)

                    table = wandb.Table(columns=["input", "pred", "overlay"])
                    for i in range(n):
                        img_np = _denorm_to_uint8(images[i], cfg['mean'], cfg['std'])           # [H,W,3] uint8
                        pmask  = (preds[i, 0].detach().cpu().numpy() > 0).astype(np.uint8) * 255

                        img_small = _resize_keep_aspect(img_np,  320, is_mask=False)
                        mask_small = _resize_keep_aspect(pmask,   320, is_mask=True)
                        overlay = _overlay_mask(img_small, mask_small > 0, color=(255, 0, 0), alpha=0.35)

                        table.add_data(
                            wandb.Image(img_small, caption = f"img {i}"),
                            wandb.Image(mask_small, caption = f"pred {i}"),
                            wandb.Image(overlay, caption = f"overlay {i}")
                        )

                        logger.log({"val/examples": table}, commit=False)
                        logged_batch = True

    epoch_loss = running_loss / len(data_loader)

    if cfg['use_wandb']:
        logger.log({"val/loss": epoch_loss}, commit=False)

    return epoch_loss

def _denorm_to_uint8(x: torch.Tensor,
                     mean:tuple = (0.485, 0.456, 0.406),
                     std:tuple = (0.229, 0.224, 0.225)) -> np.ndarray:
    '''
    Denormalizes and converts a torch image tensor to a numpy uint8 array
    Args:
        x: Image tensor of shape [3,H,W] with values in [0,1]
        mean: Mean used for normalization
        std: Standard deviation used for normalization
    Returns:
        Numpy array of shape [H,W,3] with dtype uint8
    '''
    x = x.detach().cpu()
    mean = torch.tensor(mean).view(-1, 1, 1)
    std  = torch.tensor(std).view(-1, 1, 1)
    x = (x * std + mean).clamp(0, 1)

    return (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)  # [H,W,3]

def _resize_keep_aspect(arr: np.ndarray, max_side: int, *, is_mask: bool) -> np.ndarray:
    '''
    Resize keeping aspect; images use bilinear/area, masks use nearest.
    Args:
        arr: np.ndarray of shape [H,W,3] (image) or [H,W] (mask)
        max_side: Maximum size of the longer side after resizing
        is_mask: Whether the input is a mask (True) or an image (False)
    Returns:
        Resized np.ndarray
    '''
    h, w = arr.shape[:2]

    if max(h, w) <= max_side:
        return arr
    
    scale = max_side / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pil = PILImage.fromarray(arr)
    resample = PILImage.NEAREST if is_mask else PILImage.BILINEAR

    return np.array(pil.resize((new_w, new_h), resample=resample))

def _overlay_mask(rgb_uint8: np.ndarray,
                  bin_mask: np.ndarray,
                  color=(255, 0, 0),
                  alpha: float = 0.35) -> np.ndarray:
    '''
    Overlays a binary mask on an RGB image.
    Args:
        rgb_uint8: RGB image as a numpy array of shape [H,W,3] with dtype uint8
        bin_mask: Binary mask as a numpy array of shape [H,W] with dtype bool or uint8 (0/255)
        color: Color for the mask overlay
        alpha: Transparency factor for the overlay
    Returns:
        Numpy array of shape [H,W,3] with dtype uint8 representing the overlay'''
    
    overlay = rgb_uint8.copy()
    color_layer = np.zeros_like(overlay)
    color_layer[bin_mask.astype(bool)] = color

    return (overlay * (1 - alpha) + color_layer * alpha).astype(np.uint8)
