import ipdb
import torch
import torch.nn as nn
import pytest

class DefectSegmentationLoss(nn.Module):
    def __init__(self, loss_directory:list = ['DiceLoss'], loss_type:str = 'DiceLoss'):
        super().__init__()

        if loss_type not in loss_directory:
            raise ValueError(f"loss_type must be one of {loss_directory}, got {loss_type!r}")

        self.loss_type = loss_type

        if self.loss_type == 'DiceLoss':
            self.criterion = self.dice_loss

        elif self.loss_type == 'CELoss':
            self.criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight = torch.tensor([0.4, 30.0]))

        elif self.loss_type == 'IoULoss':
            self.criterion = self.iou_loss
        
    
    def dice_loss(self, pred:torch.Tensor, target:torch.tensor, smooth:float = 1e-4)->torch.Tensor:
        """
        Computes the Dice Loss for binary segmentation.
        Args:
            pred: Tensor of predictions (batch_size, num_classes, H, W).
            target: Tensor of ground truth (batch_size, num_classes, H, W).
            smooth: Smoothing factor to avoid division by zero.
        Returns:
            Scalar Dice Loss.
        """
        # Apply sigmoid to convert logits to probabilities
        pred = torch.sigmoid(pred)
        
        # Calculate intersection and union
        intersection = (pred * target).sum(dim = (2, 3))
        union = pred.sum(dim = (2, 3)) + target.sum(dim = (2, 3)) - intersection
        
        # Compute Dice Coefficient
        dice = (intersection + smooth) / (union + smooth)
        
        # Return Dice Loss
        return 1 - dice.mean()
    
    def iou_loss(self, pred:torch.Tensor, target:torch.tensor, smooth:float = 1e-4)->torch.Tensor:
        """
        Computes the IoU Loss for binary segmentation.
        Args:
            pred: Tensor of predictions (batch_size, num_classes, H, W).
            target: Tensor of ground truth (batch_size, num_classes, H, W).
            smooth: Smoothing factor to avoid division by zero.
        Returns:
            Scalar IoU Loss.
        """
        # Apply sigmoid to convert logits to probabilities
        pred = torch.sigmoid(pred)
        
        # Calculate intersection and union
        intersection = (pred * target).sum(dim = (2, 3))
        union = pred.sum(dim = (2, 3)) + target.sum(dim = (2, 3)) - intersection
        
        # Compute IoU
        iou = (intersection + smooth) / (union + smooth)
        
        # Return IoU Loss
        return 1 - iou.mean()

    def ce_loss(self, pred:torch.Tensor, target:torch.tensor)->torch.Tensor:
        """
        Computes the Binary Cross-Entropy Loss for binary segmentation.
        Args:
            pred: Tensor of predictions (batch_size, 1, H, W).
            target: Tensor of ground truth (batch_size, 1, H, W).

        Returns:
            Scalar Cross-Entropy Loss.
        """
        target = target.to(pred.dtype)

        return self.criterion(pred, target)

    def forward(self, outputs: torch.Tensor, targets:torch.Tensor) -> torch.Tensor:
        
        '''
        Computes the loss between model outputs and ground truth targets.
        Args:
            outputs: Model predictions (logits) of shape (batch_size, num_classes, H, W).
            targets: Ground truth masks of shape (batch_size, num_classes, H, W).
        Returns:
            Computed loss value.
        '''
        if self.loss_type == 'DiceLoss':
            return self.dice_loss(outputs, targets)

        elif self.loss_type == 'CELoss':
            return self.ce_loss(outputs, targets)

        elif self.loss_type == 'IoULoss':
            return self.criterion(outputs, targets)