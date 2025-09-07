import torch.nn as nn
from model.utils import DoubleConv, DownSample, UpSample, OutConv

class Unet(nn.Module):
    def __init__(self, in_channels, num_classes = 1):
        super().__init__()

        self.down1 = DownSample(in_channels, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)

        self.outc = OutConv(64, num_classes)


    def forward(self, x):

        '''
        Forward pass through the entire model
        Args:
            x (torch.Tensor): Input tensor of shape (Batch_size, C_in, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes, H, W)
        '''
        x1, down1 = self.down1(x)      # (B, 64, H/2, W/2)
        x2, down2 = self.down2(down1)     # (B, 128, H/4, W/4)
        x3, down3 = self.down3(down2)     # (B, 256, H/8, W/8)
        x4, down4 = self.down4(down3)     # (B, 512, H/16, W/16)  

        x_bottleneck = self.bottleneck(down4)  # (B, 1024, H/16, W/16)

        x = self.up1(x_bottleneck, x4)  # (B, 512, H/8, W/8)
        x = self.up2(x, x3)             # (B, 256, H/4, W/4)
        x = self.up3(x, x2)             # (B, 128, H/2, W/2)
        x = self.up4(x, x1)             # (B, 64, H, W)

        out = self.outc(x)              # (B, num_classes, H, W)
        return out