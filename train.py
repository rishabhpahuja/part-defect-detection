import torch
from torch.utils.data import DataLoader
from loss import DefectSegmentationLoss
import wandb
import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm
from torch.cuda.amp import GradScaler

def calculate_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate Intersection over Union (IoU) for binary segmentation masks.
    
    Args:
        pred_mask: Predicted mask tensor of shape [B, 1, H, W] (logits or probabilities)
        true_mask: Ground truth mask tensor of shape [B, 1, H, W] (binary)
        threshold: Threshold for converting predictions to binary mask
    
    Returns:
        IoU score as a float
    """
    # Convert predictions to binary
    if pred_mask.max() > 1.0:  # If logits, apply sigmoid
        pred_binary = (torch.sigmoid(pred_mask) > threshold).float()
    else:  # If probabilities, just threshold
        pred_binary = (pred_mask > threshold).float()
    
    # Flatten tensors for easier computation
    pred_flat = pred_binary.view(-1)
    true_flat = true_mask.view(-1)
    
    # Calculate intersection and union
    intersection = (pred_flat * true_flat).sum()
    union = pred_flat.sum() + true_flat.sum() - intersection
    
    # Avoid division by zero
    if union == 0:
        return 1.0
    
    return (intersection / union).item()

def train(data_loader: DataLoader, model:torch.nn.Module, 
            criterion:DefectSegmentationLoss, scheduler, scaler: GradScaler,
            optimizer:torch.optim.Optimizer, device:str, mp_type: torch.dtype,
            cfg:dict, epoch:int, logger:wandb = None)->float:

    '''
    Trains the model for one epoch
    Args:
        data_loader: DataLoader for training data
        model: The neural network model
        criterion: Loss function
        scheduler: Learning rate scheduler
        optimizer: Optimizer for updating model weights
        device: Device to run the training on (CPU or GPU)
        cfg: Configuration dictionary
        epoch: Current epoch number
        logger: Wandb logger for logging metrics (optional)
    Returns:
        Average loss for the epoch
    '''

    model.train()
    running_loss = 0.0

    pbar = tqdm(data_loader, desc = f"Train Epoch:{epoch}", unit="batch")

    for i, batch in enumerate(pbar):

        images, masks = batch[0].to(device), batch[1].to(device)
        # import ipdb; ipdb.set_trace()
        optimizer.zero_grad()
        with torch.amp.autocast(dtype = mp_type, device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()
        
        pbar.set_postfix({"loss": f"{running_loss/(i+1):.4f}"})

    epoch_loss = running_loss / len(data_loader)
    
    if cfg['wandb']['use_wandb']:
        logger.log({"epoch":epoch, "train/loss": epoch_loss}, commit=False)

    return epoch_loss

def validate(data_loader: DataLoader, model:torch.nn.Module, logger:wandb,
            criterion:DefectSegmentationLoss, device:str, cfg: dict,
            epoch: int, scaler: GradScaler, mp_type: torch.dtype)->float:

    '''
    Validates the model for one epoch
    Args:
        data_loader: DataLoader for validation data
        model: The neural network model
        logger: Wandb logger for logging metrics
        criterion: Loss function
        device: Device to run the validation on (CPU or GPU)
        cfg: Configuration dictionary
        epoch: Current epoch number
        scaler: Gradient scaler for mixed precision
        mp_type: Mixed precision data type
    Returns:
        Average loss for the epoch (IoU is logged to wandb)
    '''

    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    logged_batch = False

    pbar = tqdm(data_loader, desc = f"Val Epoch:{epoch}", unit="batch")
    table = wandb.Table(columns=["epoch", "input", "GT", "pred", "overlay"])

    with torch.no_grad():
        for i, batch in enumerate(data_loader):

            images, masks = batch[0].to(device), batch[1].to(device)
            with torch.amp.autocast(dtype = mp_type, device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, masks)

            # Calculate IoU for this batch
            batch_iou = calculate_iou(outputs, masks, cfg['data']['class_threshold'])
            
            running_loss += loss.item()
            running_iou += batch_iou
            
            pbar.set_postfix({
                "loss": f"{running_loss/(i+1):.4f}",
                "iou": f"{running_iou/(i+1):.4f}"
            })

            if cfg['wandb']['use_wandb']:
                # Log a few validation examples from the 1st batch to wandb
                if not logged_batch:
                    probs = torch.sigmoid(outputs)                 
                    preds = (probs > cfg['data']['class_threshold']).float()
                    B = images.size(0)
                    n = min(3, B)

                    for i in range(n):
                        img_np = _denorm_to_uint8(images[i], cfg['data']['mean'], cfg['data']['std'])           # [H,W,3] uint8
                        pmask  = (preds[i, 0].detach().cpu().numpy() > 0).astype(np.uint8) * 255

                        img_small = _resize_keep_aspect(img_np,  320, is_mask=False)
                        gt_small = _resize_keep_aspect(masks[i,0].detach().cpu().numpy().astype(np.uint8)*255, 320, is_mask=True)
                        mask_small = _resize_keep_aspect(pmask,   320, is_mask=True)
                        overlay = _overlay_mask(img_small, mask_small > 0, color=(255, 0, 0), alpha=0.35)

                        table.add_data(
                            epoch,
                            wandb.Image(img_small, caption = f"img {i}"),
                            wandb.Image(gt_small, caption = f"GT {i}"),
                            wandb.Image(mask_small, caption = f"pred {i}"),
                            wandb.Image(overlay, caption = f"overlay {i}")
                        )

                    logger.log({"epoch": epoch, "val/examples": table}, step = epoch, commit=False)
                    logged_batch = True

    epoch_loss = running_loss / len(data_loader)
    epoch_iou = running_iou / len(data_loader)

    if cfg['wandb']['use_wandb']:
        logger.log({"epoch": epoch, 
                    "val/loss": epoch_loss,
                    "val/iou": epoch_iou
                    }, commit=False)

    return epoch_loss, epoch_iou

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
