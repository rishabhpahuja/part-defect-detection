import torch
import torch.nn as nn
import pytest

_ALLOWED = {'DiceLoss'}

class DefectSegmentationLoss(nn.Module):
    def __init__(self, loss_type = 'DiceLoss'):
        super().__init__()

        if loss_type not in _ALLOWED:
            raise ValueError(f"loss_type must be one of {_ALLOWED}, got {loss_type!r}")

        self.criterion = loss_type
    
    def dice_loss(self, pred:torch.Tensor, target:torch.tensor, smooth:float = 1e-4)->torch.Tensor:
        """
        Computes the Dice Loss for binary segmentation.
        Args:
            pred: Tensor of predictions (batch_size, 1, H, W).
            target: Tensor of ground truth (batch_size, 1, H, W).
            smooth: Smoothing factor to avoid division by zero.
        Returns:
            Scalar Dice Loss.
        """
        # Apply sigmoid to convert logits to probabilities
        pred = torch.sigmoid(pred)
        
        # Calculate intersection and union
        intersection = (pred * target).sum(dim = (2, 3))
        union = pred.sum(dim = (2, 3)) + target.sum(dim = (2, 3))
        
        # Compute Dice Coefficient
        dice = (2. * intersection + smooth) / (union + smooth)
        
        # Return Dice Loss
        return 1 - dice.mean()

    def forward(self, outputs: torch.Tensor, targets:torch.Tensor) -> torch.Tensor:
        
        '''
        Computes the loss between model outputs and ground truth targets.
        Args:
            outputs: Model predictions (logits) of shape (batch_size, 1, H, W).
            targets: Ground truth masks of shape (batch_size, 1, H, W).
        Returns:
            Computed loss value.
        '''
        if self.criterion == 'DiceLoss':
            return self.dice_loss(outputs, targets)