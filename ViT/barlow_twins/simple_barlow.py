"""
Updated Barlow Twins SSL Training Script 
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import logging

# Import our custom model and dataset
from fixed_model import BarlowTwinsModel
from simple_h52 import SimpleH5Dataset, get_barlow_twins_transforms
from transformers import get_cosine_schedule_with_warmup

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class BarlowTwinsLoss(nn.Module):
    """
    Barlow Twins loss function
    """
    def __init__(self, lambda_coeff=5e-3):
        super().__init__()
        self.lambda_coeff = lambda_coeff
    
    def forward(self, z1, z2):
    # Normalize projections along the batch dimension
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        
        std1 = torch.sqrt(z1.var(dim=0, unbiased=False) + 1e-6)
        std2 = torch.sqrt(z2.var(dim=0, unbiased=False) + 1e-6)
        
        z1 = z1 / std1
        z2 = z2 / std2
        
        batch_size = z1.size(0)
        feature_dim = z1.size(1)
        
        # Cross-correlation matrix
        c = torch.matmul(z1.T, z2) / batch_size
        
        # Loss calculation using matrix operations
        identity = torch.eye(feature_dim, device=c.device)
        c_diff = c - identity
        
        # Create a mask for off-diagonal elements
        off_diagonal_mask = ~torch.eye(feature_dim, dtype=bool, device=c.device)
        
        # Apply different weights to diagonal vs off-diagonal
        c_diff[off_diagonal_mask] *= self.lambda_coeff
        
        # Calculate loss as sum of squared differences
        loss = (c_diff ** 2).sum()
        
        # Store components for logging
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = torch.sum(c[off_diagonal_mask] ** 2)
        
        self.last_on_diag = on_diag.item()
        self.last_off_diag = off_diag.item()
        self.last_weighted_off_diag = (self.lambda_coeff * off_diag).item()
        
        return loss

    def _off_diagonal(self, x):
        n = x.size(0)
        return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()

def train_barlow_twins(
    train_h5_path,
    output_dir='./output',
    batch_size=16,
    num_epochs=100,
    lr=5e-5,
    weight_decay=0.04,
    lambda_coeff=0.005,
    patch_size=224,
    vit_patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
    projection_dim=512,
    num_workers=4,
    device=None
):
    """
    Train a Vision Transformer with Barlow Twins on histopathology data
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    logger.info(f"Training with batch size: {batch_size} for {num_epochs} epochs")
    
    # Create model with our fixed architecture
    model = BarlowTwinsModel(
        image_size=patch_size,
        patch_size=vit_patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        proj_dim=projection_dim
    ).to(device)
    
    # Create loss function
    criterion = BarlowTwinsLoss(lambda_coeff=lambda_coeff)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # Get transforms for two views
    transform1, transform2 = get_barlow_twins_transforms(patch_size=patch_size)
    
    # Create datasets for the two views
    dataset1 = SimpleH5Dataset(train_h5_path, transform=transform1)
    dataset2 = SimpleH5Dataset(train_h5_path, transform=transform2)
    
    # Create data loaders
    loader1 = DataLoader(
        dataset1,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    loader2 = DataLoader(
        dataset2,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Make sure both loaders have the same number of batches
    assert len(loader1) == len(loader2)
    
    warmup_epochs = 10  # You can tune this
    total_steps = len(loader1) * num_epochs
    warmup_steps = len(loader1) * warmup_epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
        )

    # Initialize best loss for saving best model
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_on_diag = 0.0
        running_off_diag = 0.0
        running_weighted_off_diag = 0.0
        
        # Use zip to iterate through both loaders simultaneously
        for batch_idx, (images1, images2) in enumerate(zip(loader1, loader2)):
            # Move images to device
            images1 = images1.to(device)
            images2 = images2.to(device)

            # Small noise to images can help prevent perfect alignment
            if epoch < 10:  # Only during early epochs
                images1 = images1 + 0.01 * torch.randn_like(images1)
                images2 = images2 + 0.01 * torch.randn_like(images2)
            
            # Forward pass
            z1 = model(images1)
            z2 = model(images2)
            
            # Calculate loss
            loss = criterion(z1, z2)

            # Track loss components
            running_on_diag += criterion.last_on_diag
            running_off_diag += criterion.last_off_diag
            running_weighted_off_diag += criterion.last_weighted_off_diag
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Optional: Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            
            # Print progress
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(loader1)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"On-diag: {criterion.last_on_diag:.4f}, "
                    f"Off-diag: {criterion.last_off_diag:.4f}, "
                    f"Weighted Off-diag: {criterion.last_weighted_off_diag:.4f}"
                    )
        
        # Update scheduler
        scheduler.step()
        
        # Calculate epoch averages
        epoch_loss = running_loss / len(loader1)
        epoch_on_diag = running_on_diag / len(loader1)
        epoch_off_diag = running_off_diag / len(loader1)
        epoch_weighted_off_diag = running_weighted_off_diag / len(loader1)
        
        # Log epoch results with components
        logger.info(
            f"Epoch {epoch} completed, "
            f"Loss: {epoch_loss:.4f}, "
            f"On-diag: {epoch_on_diag:.4f}, "
            f"Off-diag: {epoch_off_diag:.4f}, "
            f"Weighted Off-diag: {epoch_weighted_off_diag:.4f}, "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        
        # Save best model
        if epoch == 0 or epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(output_dir, 'model_best.pt'))
    
    logger.info("Training completed!")
    return model

def main():
    """Main function to run training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Barlow Twins with Vision Transformer')
    parser.add_argument('--train_h5_path', type=str, required=True,
                        help='Path to training H5 file')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for saving models')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.04,
                        help='Weight decay')
    parser.add_argument('--lambda_coeff', type=float, default=5e-3,
                        help='Lambda coefficient for Barlow Twins')
    parser.add_argument('--patch_size', type=int, default=224,
                        help='Patch size for images')
    parser.add_argument('--vit_patch_size', type=int, default=16,
                        help='Patch size for ViT')
    parser.add_argument('--embed_dim', type=int, default=768,
                        help='Embedding dimension for ViT')
    parser.add_argument('--depth', type=int, default=12,
                        help='Depth of ViT')
    parser.add_argument('--num_heads', type=int, default=12,
                        help='Number of attention heads in ViT')
    parser.add_argument('--projection_dim', type=int, default=8192,
                        help='Projection dimension for Barlow Twins')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Train model
    train_barlow_twins(
        train_h5_path=args.train_h5_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_coeff=args.lambda_coeff,
        patch_size=args.patch_size,
        vit_patch_size=args.vit_patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        projection_dim=args.projection_dim,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main()
