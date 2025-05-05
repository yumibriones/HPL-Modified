"""
Barlow Twins Self-Supervised Learning with Vision Transformer for Histopathology Images

This implementation includes:
1. Dataset preparation and augmentation for SSL
2. Barlow Twins contrastive learning implementation
3. Vision Transformer architecture
4. Distributed training for HPC
5. Experiment tracking
"""

import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import time
import math
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
from torchvision.models.vision_transformer import VisionTransformer, EncoderBlock

from PIL import Image, ImageFilter, ImageOps

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

#############################################
# Dataset and Augmentation Implementation
#############################################

class GaussianBlur:
    """Gaussian blur augmentation from SimCLR paper"""
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class HistoPathologyDataset(Dataset):
    """Dataset for histopathology patches with SSL augmentations"""
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with all the image patches
            transform (callable, optional): Optional transform to be applied on each image
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        # Modified to work with organized train directory
        self.samples = list(self.data_dir.glob('**/*.png')) + list(self.data_dir.glob('**/*.jpg'))
        logger.info(f"Found {len(self.samples)} image samples in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            return self.transform(image)
        return image

class TwoCropsTransform:
    """Create two crops of the same image"""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

def get_ssl_transforms(patch_size=224):
    """
    Returns the transformations for SSL training
    with strong augmentations for contrastive learning
    """
    # Define augmentation pipeline following SimCLR/Barlow Twins approaches
    # but adapted for histopathology characteristics
    color_jitter = transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
    )
    
    normalize = transforms.Normalize(
        mean=[0.7, 0.6, 0.7],  # Adjusted for typical H&E stained histopathology
        std=[0.15, 0.15, 0.15]
    )
    
    # SSL augmentation pipeline
    transform = transforms.Compose([
        transforms.RandomResizedCrop(patch_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  # Histopathology is orientation-invariant
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.ToTensor(),
        normalize,
    ])
    
    return TwoCropsTransform(transform)

#############################################
# Vision Transformer Implementation
#############################################

class BarlowTwinsVisionTransformer(nn.Module):
    """
    Vision Transformer with Barlow Twins projection head
    """
    def __init__(
        self, 
        image_size=224, 
        patch_size=16, 
        in_channels=3, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4.0, 
        proj_dim=8192, 
        drop_rate=0.0,
        attn_drop_rate=0.0
    ):
        super().__init__()
        
        # ViT encoder
        self.encoder = VisionTransformer(
            image_size=image_size, 
            patch_size=patch_size,
            num_layers=depth,
            num_heads=num_heads,
            hidden_dim=embed_dim,
            mlp_dim=int(embed_dim * mlp_ratio),
            dropout=drop_rate,
            attention_dropout=attn_drop_rate,
            num_classes=0  # No classification head
        )
        
        # Barlow Twins projection head
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, proj_dim),
        )
        
    def forward(self, x):
        # Get the encoder output (CLS token)
        features = self.encoder(x)
        
        # Apply projector
        z = self.projector(features)
        
        # Normalize projections
        z = F.normalize(z, dim=1)
        
        return z

#############################################
# Barlow Twins Loss Implementation
#############################################

class BarlowTwinsLoss(nn.Module):
    """
    Implementation of Barlow Twins loss function
    """
    def __init__(self, lambda_coeff=5e-3, batch_size=128, world_size=1):
        super().__init__()
        self.lambda_coeff = lambda_coeff
        self.batch_size = batch_size
        self.world_size = world_size

    def forward(self, z1, z2):
        # normalize the representations along the batch dimension
        batch_size = z1.size(0)
        feature_dim = z1.size(1)
        
        # Calculate cross-correlation matrix
        c = torch.matmul(z1.T, z2) / batch_size
        
        # Calculate the loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self._off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambda_coeff * off_diag
        
        return loss

    def _off_diagonal(self, x):
        # Returns the off-diagonal elements of a square matrix
        n = x.size(0)
        return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()

#############################################
# Training Functions
#############################################

def train_one_epoch(model, data_loader, optimizer, criterion, epoch, device, args):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    
    progress = tqdm(data_loader, desc=f"Epoch {epoch}")
    end = time.time()
    
    model.train()
    for i, images in enumerate(progress):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # Each image is a list of two views
        images[0] = images[0].to(device, non_blocking=True)
        images[1] = images[1].to(device, non_blocking=True)
        
        # compute output and loss
        z1 = model(images[0])
        z2 = model(images[1])
        
        loss = criterion(z1, z2)
        
        # record loss
        losses.update(loss.item(), images[0].size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        progress.set_postfix({'Loss': losses.avg})
    
    return losses.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def save_checkpoint(state, is_best, filename='checkpoint.pt', best_filename='model_best.pt'):
    """Save checkpoint to disk"""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)

#############################################
# Distributed Training Setup
#############################################

def setup(rank, world_size, port):
    """Set up distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()

def train_ssl(rank, world_size, args):
    """Main training function for each process"""
    # Set up distributed training
    setup(rank, world_size, args.port)
    
    # Set random seeds for reproducibility
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    
    # Create model directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create TensorBoard writer if rank 0
    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # Create model
    logger.info(f"Creating model: BarlowTwinsVisionTransformer")
    model = BarlowTwinsVisionTransformer(
        image_size=args.patch_size, 
        patch_size=args.vit_patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        proj_dim=args.projection_dim
    )
    
    # Move model to GPU
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Define loss function
    criterion = BarlowTwinsLoss(
        lambda_coeff=args.lambda_coeff,
        batch_size=args.batch_size,
        world_size=world_size
    )
    
    # Define optimizer and learning rate scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr * args.batch_size * world_size / 256,
        weight_decay=args.weight_decay
    )
    
    # Cosine decay learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # SSL transformations
    transform = get_ssl_transforms(patch_size=args.patch_size)
    
    # Create dataset and dataloader
    train_dataset = HistoPathologyDataset(
        data_dir=args.data_dir,
        transform=transform
    )
    
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            device=rank,
            args=args
        )
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        if rank == 0:
            logger.info(f"Epoch {epoch}: Loss={train_loss:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")
            
            # Write to TensorBoard
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)
            
            # Save checkpoint
            is_best = train_loss < best_loss
            best_loss = min(train_loss, best_loss)
            
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_loss': best_loss,
            }, is_best, 
            filename=os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pt'),
            best_filename=os.path.join(args.output_dir, 'model_best.pt'))
    
    # Clean up distributed training
    cleanup()
    
    # Close TensorBoard writer
    if rank == 0 and writer is not None:
        writer.close()

#############################################
# Main script
#############################################

def main():
    parser = argparse.ArgumentParser(description='Barlow Twins SSL with Vision Transformer')
    # Dataset parameters
    parser.add_argument('--data-dir', default='/path/to/histopathology/patches', type=str,
                        help='path to dataset')
    parser.add_argument('--output-dir', default='./output', type=str,
                        help='path where to save outputs')
    
    # Training parameters
    parser.add_argument('--epochs', default=100, type=int, 
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=64, type=int, 
                        help='mini-batch size per GPU')
    parser.add_argument('--lr', default=3e-4, type=float, 
                        help='base learning rate')
    parser.add_argument('--weight-decay', default=0.04, type=float, 
                        help='weight decay')
    parser.add_argument('--lambda-coeff', default=5e-3, type=float, 
                        help='lambda coefficient for the off-diagonal terms in Barlow Twins')
    
    # Model parameters
    parser.add_argument('--patch-size', default=224, type=int, 
                        help='input image size')
    parser.add_argument('--vit-patch-size', default=16, type=int, 
                        help='patch size for ViT')
    parser.add_argument('--embed-dim', default=768, type=int, 
                        help='embedding dimension for ViT')
    parser.add_argument('--depth', default=12, type=int, 
                        help='depth of transformer layers')
    parser.add_argument('--num-heads', default=12, type=int, 
                        help='number of attention heads')
    parser.add_argument('--mlp-ratio', default=4.0, type=float, 
                        help='ratio of mlp hidden dim to embedding dim')
    parser.add_argument('--projection-dim', default=8192, type=int, 
                        help='projection dimension for Barlow Twins')
    
    # Distributed training parameters
    parser.add_argument('--nodes', default=1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--gpus-per-node', default=4, type=int, 
                        help='number of GPUs per node')
    parser.add_argument('--node-rank', default=0, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--port', default=29500, type=int, 
                        help='port for distributed training')
    
    # Misc
    parser.add_argument('--workers', default=8, type=int, 
                        help='number of data loading workers')
    parser.add_argument('--seed', default=42, type=int, 
                        help='random seed')
    
    args = parser.parse_args()
    
    # Calculate world size
    world_size = args.nodes * args.gpus_per_node
    
    # Use torch.multiprocessing.spawn to launch distributed processes
    mp.spawn(
        train_ssl,
        args=(world_size, args),
        nprocs=args.gpus_per_node,
        join=True
    )

if __name__ == "__main__":
    import shutil
    main()
