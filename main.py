import torch
from torch.utils.data import DataLoader
import wandb
import yaml
import os

from loss import DefectSegmentationLoss
from model.model import Unet
from train import train, validate
from dataloader import DefectDataset

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
        wandb.init(project = cfg['wandb']['project'], name = f"_{cfg['logger_comment']}\
                                _bs{cfg['train']['batch_size']}_\
                                lr{cfg['train']['learning_rate']}\
                                _wd{cfg['train']['weight_decay']}")
    
    # Create datasets and dataloaders
    dataset = DefectDataset(img_dir=cfg['data']['train_dir'], 
                                  num_classes = cfg['data']['num_classes'], 
                                  img_size = cfg['data']['img_size'], 
                                  train=True, 
                                  cfg = cfg)
    
    # train validation dataset split
    val_size = int(len(dataset) * (1 - cfg['train']['test_train_split']))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, 
                                                               [train_size, val_size])  
    
    train_loader = DataLoader(train_dataset, 
                              batch_size = cfg['train']['batch_size'], 
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, 
                            batch_size=cfg['train']['batch_size'], 
                            shuffle=False, num_workers=4)
    
    # Initialize model
    model = Unet(in_channels = 3, num_classes = cfg['data']['num_classes']).to(device)
    
    # Initialize loss function
    criterion = DefectSegmentationLoss(loss_type=cfg['loss']['type'],
                                       loss_directory = cfg['loss']['loss_types']).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['learning_rate'], 
                                  weight_decay=cfg['train']['weight_decay'])
    
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

    # Training loop
    least_val_loss = float('inf')
    for epoch in range(cfg['train']['num_epochs']):
        train_loss = train(data_loader=train_loader, model=model, criterion=criterion,
                           optimizer=optimizer, device=device,
                           scheduler=None, epoch=epoch, cfg=cfg)
        
        val_loss = validate(data_loader=val_loader, model=model, criterion=criterion,
                            device=device, cfg=cfg)
        # Save model if it has the least validation loss so far
        if cfg['train']['save_weights'] and val_loss < least_val_loss:
            least_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(experiment_dir, 'best_model.pth'))
            print(f"Saved best model with val loss: {least_val_loss:.4f} at epoch {epoch+1}")
        
        if cfg['wandb']['use_wandb']:
            wandb.log({'Train Loss': train_loss, 'Validation Loss': val_loss}, step=epoch)
        
        print(f"Epoch [{epoch+1}/{cfg['train']['num_epochs']}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    # Load configuration from YAML file
    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    
    main(cfg)