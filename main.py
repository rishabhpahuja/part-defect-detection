import torch
from torch.utils.data import DataLoader, Subset
import wandb
import yaml
import os

from loss import DefectSegmentationLoss
from model.model import Unet
from train import train, validate
from dataloader import DefectDataset, visualize_train_samples

def construct_dataloaders(cfg):
    '''
    Constructs training and validation dataloaders
    Args:
        cfg: Configuration dictionary containing data parameters
    '''
    dataset = DefectDataset(img_dir=cfg['train']['train_dir'], 
                                  num_classes = cfg['data']['num_classes'], 
                                  img_size = cfg['data']['img_size'],  
                                  cfg = cfg)
    
    # Visualize some training samples
    visualize_train_samples(dataset, n = 8, mean = cfg['data']['mean'], std = cfg['data']['std'])

    g = torch.Generator().manual_seed(cfg["seed"])
    va_frac = 1 - cfg['train']['test_train_split']
    val_size = int(len(dataset) * va_frac)
    train_size = len(dataset) - val_size
    train_idx, val_idx = torch.utils.data.random_split(dataset, 
                                                               [train_size, val_size],
                                                               generator=g)

    train_ds_full = DefectDataset(img_dir=cfg["train"]["train_dir"],
        num_classes = cfg["data"]["num_classes"],
        img_size = cfg["data"]["img_size"],
        cfg=cfg,
    )
    val_ds_full = DefectDataset(
        img_dir = cfg["train"]["train_dir"],
        num_classes = cfg["data"]["num_classes"],
        img_size = cfg["data"]["img_size"],
        val = True,
        cfg = cfg,
    )

    train_dataset = Subset(train_ds_full, train_idx.indices if hasattr(train_idx, "indices") else train_idx)
    val_dataset   = Subset(val_ds_full,   val_idx.indices   if hasattr(val_idx,   "indices")   else val_idx)

    train_loader = DataLoader(train_dataset, batch_size=cfg["train"]["batch_size"], shuffle=True,  num_workers=cfg["train"]["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"], pin_memory=True)

    return train_loader, val_loader

def main(cfg):
    '''
    Main function to set up data, model, loss function, optimizer, and start training.
    Args:
        cfg: Configuration dictionary containing all necessary parameters.
    '''
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize WandB if enabled
    if cfg['wandb']['use_wandb']:
        wandb.init(project = cfg['wandb']['project'], name = f"experiment_{len(os.listdir(cfg['train']['model_save_dir'])) + 1}_\
                   {cfg['logger_comment']}_bs{cfg['train']['batch_size']}_\
                   lr{cfg['train']['learning_rate']}_\
                   wd{cfg['train']['weight_decay']}")
    
    # Create datasets and dataloaders
    train_loader, val_loader = construct_dataloaders(cfg)

    # Initialize model
    if cfg['use_saved_model']:
        model = Unet(in_channels = 3, num_classes = cfg['data']['num_classes']).to(device)
        model.load_state_dict(torch.load(cfg['saved_model_path'], map_location=device))
        print(f"Loaded model weights from {cfg['saved_model_path']}")
    else:
        model = Unet(in_channels = 3, num_classes = cfg['data']['num_classes']).to(device)

    # Initialize loss function
    criterion = DefectSegmentationLoss(loss_type=cfg['loss']['type'],
                                       loss_directory = cfg['loss']['loss_types']).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['learning_rate'], 
                                  weight_decay= float(cfg['train']['weight_decay']))
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.9)

    # Define scalar for mixed precision training
    scaler = torch.amp.GradScaler()
    
    if cfg['train']['save_weights']:
        # Create directory to save models if it doesn't exists
        os.makedirs(cfg['train']['model_save_dir'], exist_ok=True)
        # Create a directory to save experiment weights
        folder_name = f"experiment_{len(os.listdir(cfg['train']['model_save_dir'])) + 1}" + \
                        f"_bs{cfg['train']['batch_size']}_lr{cfg['train']['learning_rate']}" + \
                        f"_wd_{cfg['train']['weight_decay']}"+f"_{cfg['logger_comment']}"
        experiment_dir = os.path.join(cfg['train']['model_save_dir'], folder_name)
        os.makedirs(experiment_dir)
        # Copy model, train, cfg file to experiment directory
        os.system(f"cp model/model.py {experiment_dir}")
        os.system(f"cp train.py {experiment_dir}")
        os.system(f"cp config.yaml {experiment_dir}/config.yaml")
        print(f"Experiment weights and code will be saved to {experiment_dir}")

    # Mixed Precision data type
    mp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Training loop
    least_val_loss = float('inf')
    max_iou = 0.0
    for epoch in range(cfg['train']['epochs']):
        train_loss = train(data_loader=train_loader, model=model, criterion=criterion,
                           optimizer=optimizer, device=device, mp_type = mp_dtype,
                           scheduler=scheduler, epoch=epoch, cfg=cfg, scaler = scaler,
                           logger=wandb if cfg['wandb']['use_wandb'] else None)
        
        val_loss, iou = validate(data_loader=val_loader, model=model, criterion=criterion,
                            device=device, cfg=cfg, epoch=epoch, scaler = scaler, mp_type = mp_dtype,
                            logger=wandb if cfg['wandb']['use_wandb'] else None)
        # Save model if it has the least validation loss so far
        if cfg['train']['save_weights'] and iou > max_iou:
            max_iou = iou
            torch.save(model.state_dict(), os.path.join(experiment_dir, 'best_model.pth'))
            print(f"Saved best model with val loss: {least_val_loss:.4f} at epoch {epoch+1}")
        
        if cfg['wandb']['use_wandb']:
            wandb.log({'Train Loss': train_loss, 'Validation Loss': val_loss}, step=epoch)

        print(f"Epoch [{epoch+1}/{cfg['train']['epochs']}], Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, IoU: {iou:.4f}\n")

if __name__ == "__main__":
    # Load configuration from YAML file
    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    
    main(cfg)