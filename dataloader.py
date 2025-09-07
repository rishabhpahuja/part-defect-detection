import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.tv_tensors import Image, Mask
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
import os


class DefectDataset(Dataset):
    def __init__(self, img_dir: str, num_classes:int = 1,
                 img_size:int = 640,  train:bool = True, cfg:dict = None):

        '''
        img_dir: Directory containing images and masks (if train=True). e.g. 'data/test/white_bracket'
        img_size: Desired size to which images and masks will be resized (img_size x img_size)
        train: Boolean indicating whether the dataset is for training (True) and validation, False fortesting
        '''

        self.train = train
        self.img_size = img_size
        self.num_classes = num_classes
        self.images = []
        if self.train:
            self.masks = []

        for directory in os.listdir(img_dir):

            directory_path = os.path.join(img_dir, directory) # e.g. 'test/white_bracket/hole_defect'

            for image in os.listdir(directory_path):

                if image.endswith('.png'):    
                    self.images.append(os.path.join(img_dir, directory, image)) # e.g. 'test/white_bracket/hole_defect/img1.png'

                    if self.train:
                        if directory != 'good': #Since good images don't have masks
                            self.masks.append(os.path.join(img_dir, 'ground_truth', directory, image)) # e.g. 'test/white_bracket/ground_truth/hole_defect/img1.png'
                        else:
                            self.masks.append('empty') #Placeholder for good images

        if self.train:

            assert len(self.images) == len(self.masks), "Number of images and masks should be the same in training mode"

            self.train_transform = transforms.Compose([transforms.ToImage(),
                                                        transforms.ToDtype(torch.float32, scale = True),
                                                        transforms.RandomHorizontalFlip(p = 0.5),
                                                        transforms.RandomVerticalFlip(p = 0.5),
                                                        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                                                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                                        transforms.Resize((self.img_size, self.img_size)),
                                                        transforms.Normalize(mean = cfg['data']['mean'],\
                                                                            std = cfg['data']['mean']), # Placeholder transforms for training images
                                        ]) 
            

        self.val_transform = transforms.Compose([
                                                transforms.ToImage(),
                                                transforms.ToDtype(torch.float32, scale = True),
                                                transforms.Resize((img_size, img_size)),
                                                transforms.Normalize(mean = cfg['data']['mean'],
                                                                    std=cfg['data']['mean']), # Placeholder transforms for validation images
                                        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.train:
            mask_path = self.masks[idx]
            if mask_path == 'empty':
                mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.uint8) # Create an empty mask for good images
            else:
                mask = Image.open(mask_path).convert('L')

            image, mask = Image(image), Mask(mask)
            image, mask = self.train_transform(image, mask)

            return image, mask

        else:
            image = Image(image)
            image = self.val_transform(image)    
    
            return image

# Unnormalize helper
def _unnormalize(img: torch.Tensor, mean: tuple = (0.485, 0.456, 0.406), std:tuple = (0.229, 0.224, 0.225)):
    # img: [3,H,W] float tensor
    mean = torch.tensor(mean, device=img.device).view(3, 1, 1)
    std = torch.tensor(std, device=img.device).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)

@torch.no_grad()
def visualize_train_samples(dataset: DataLoader, n: int = 8, mean:tuple = (0.485,0.456,0.406), std:tuple = (0.229,0.224,0.225)):
    """
    Plots a 2×n grid (row1: images, row2: corresponding masks) using first n samples.
    Works with your DefectDataset(train=True).
    """
    n = min(n, len(dataset))
    imgs, masks = [], []
    for i in range(n):
        img, msk, _ = dataset[i]  # [3,H,W], [H,W]
        imgs.append(_unnormalize(img, mean, std).cpu())
        masks.append(msk.cpu())

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
    plt.show()