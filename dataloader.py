import torch
import matplotlib.pyplot as plt
from torchvision.tv_tensors import Image, Mask
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms
import os
from PIL import Image as PILImage


class DefectDataset(Dataset):
    def __init__(self, img_dir: str, num_classes:int = 1,
                 img_size:int = 640, train:bool = True, val:bool = False,
                 cfg:dict = None):

        '''
        img_dir: Directory containing images and masks (if train=True). e.g. 'data/test/white_bracket'
        img_size: Desired size to which images and masks will be resized (img_size x img_size)
        train: Boolean indicating whether the dataset is for training (True) and validation, False for testing
        val: Boolean indicating whether the dataset is for validation (True) or training (False)
        '''

        self.train = train
        self.val = val
        self.img_size = img_size
        self.num_classes = num_classes
        self.images = []
        if self.train:
            self.masks = []

        for directory in os.listdir(img_dir):

            directory_path = os.path.join(img_dir, directory) # e.g. 'white_bracket/test/hole_defect'

            for image in os.listdir(directory_path):

                if image.endswith('.png'):    
                    self.images.append(os.path.join(img_dir, directory, image)) # e.g. 'white_bracket/test/hole_defect/img1.png'

                    if self.train:
                        if directory != 'good': #Since good images don't have masks
                            self.masks.append(os.path.join(img_dir.split('/')[0], 'ground_truth', 
                                                    directory, image.split('.')[0] + '_mask.png'   )) # e.g. 'white_bracket/test/ground_truth/hole_defect/img1_mask.png'
                        else:
                            self.masks.append('empty') #Placeholder for good images

        if self.val:
            self.transform = transforms.Compose([
                                                    transforms.ToImage(),
                                                    transforms.ToDtype(torch.float32, scale = True),
                                                    transforms.Resize((img_size, img_size)),
                                                    transforms.Normalize(mean = cfg['data']['mean'],
                                                                        std = cfg['data']['std']), 
                                        ])
        elif self.train:
            assert len(self.images) == len(self.masks), "Number of images and masks should be the same in training mode"
            self.transform = transforms.Compose([transforms.ToImage(),
                                                        transforms.ToDtype(torch.float32, scale = True),
                                                        transforms.RandomHorizontalFlip(p = 0.5),
                                                        transforms.RandomVerticalFlip(p = 0.5),
                                                        transforms.RandomEqualize(p = 0.5),
                                                        transforms.GaussianBlur(kernel_size=(3, 3)),
                                                        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                                                        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                                        transforms.Resize((self.img_size, self.img_size)),
                                                        transforms.Normalize(mean = cfg['data']['mean'],\
                                                                            std = cfg['data']['std']), 
                                        ]) 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]
        image = PILImage.open(img_path).convert('RGB')
        
        if self.train:
            mask_path = self.masks[idx]
            if mask_path == 'empty':
                mask = torch.zeros((1, image.size[1], image.size[0]), dtype=torch.uint8) # Create an empty mask for good images
            else:
                mask = PILImage.open(mask_path).convert('L')

            image, mask = Image(image), Mask(mask)
            image, mask = self.transform(image, mask)
            mask = (mask > 0).to(torch.float32)
            
            return image, mask

        else:
            image = Image(image)
            image = self.transform(image)    
    
            return image

def _unnormalize(img: torch.Tensor, mean: tuple, std:tuple):
    '''
    Denormalizes an image tensor using the provided mean and std.
    Args:
        img: Normalized image tensor of shape [3,H,W] with values in [0,1]
        mean: Mean used for normalization
        std: Standard deviation used for normalization
    Returns:
        Denormalized image tensor of shape [3,H,W] with values in [0,1]
    '''
    mean = torch.tensor(mean, device=img.device).view(-1, 1, 1)
    std = torch.tensor(std, device=img.device).view(-1, 1, 1)
    return (img * std + mean).clamp(0, 1)

@torch.no_grad()
def visualize_train_samples(dataset: DefectDataset, n: int = 8, mean:tuple = None, 
                            std:tuple = None):
    """
    Plots a 2×n grid (row1: images, row2: corresponding masks) using first n samples.
    Works with your DefectDataset(train=True).
    """
    n = min(n, len(dataset))
    imgs, masks = [], []
    for i in range(n):
        img, msk = dataset[-i]  # [3,H,W], [H,W]
        imgs.append(_unnormalize(img, mean = mean, std = std).cpu())
        masks.append(msk.squeeze().cpu())

    H = 2
    W = n
    fig, axes = plt.subplots(H, W, figsize=(2.3*W, 2.3*H))

    for j in range(W):
        # row 0: image
        ax0 = axes[0, j] if W > 1 else axes[0]
        ax0.imshow(imgs[j].permute(1, 2, 0).numpy())
        ax0.set_title(f"img {j}")
        ax0.axis("off")

        # row 1: mask
        ax1 = axes[1, j] if W > 1 else axes[1]
        ax1.imshow(masks[j].numpy(), cmap="gray", vmin=0, vmax=masks[j].max().item() or 1)
        ax1.set_title(f"mask {j}")
        ax1.axis("off")

    plt.tight_layout()
    plt.savefig('train_samples.png', dpi=300)
    plt.close()