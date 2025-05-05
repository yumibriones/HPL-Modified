"""
DINO Model Implementation for Self-Supervised Learning on Histopathology Data
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
from torchvision.models.vision_transformer import VisionTransformer

from torch.amp import autocast as autocast_cuda
from torch.amp import GradScaler

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

# Multi-Crop Wrapper for DINO
class MultiCropWrapper(nn.Module):
    """
    Wrapper to handle multiple crops for DINO
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # Backbone network (ViT)
        self.backbone = backbone
        # Projection head
        self.head = head
        
    def forward(self, x):
    # Convert list of tensors to single tensor if needed
        if isinstance(x, list):
            # Special case: list with single element
            if len(x) == 1:
                return self.head(self.backbone(x[0]))
            
            # Multiple crops case
            idx_crops = torch.cumsum(torch.unique_consecutive(
                torch.tensor([inp.shape[0] for inp in x]),
                return_counts=True,
            )[1], 0)
            
            start_idx = 0
            features = []
            for end_idx in idx_crops:
                _features = torch.cat([self.backbone(x[i]) for i in range(start_idx, end_idx)])
                features.append(_features)
                start_idx = end_idx
            
            # Apply head to all features
            return [self.head(f) for f in features]
        else:
            return self.head(self.backbone(x))

# DINO Head Network
class DINOHead(nn.Module):
    """
    DINO projection head
    """
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256, use_bn=False):
        super(DINOHead, self).__init__()
        
        # Implement MLP with optional BN
        if use_bn:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, bottleneck_dim),
            )
        
        # Final projection - use regular linear layer instead
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
                
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

# Feature Extractor based on ViT
class FeatureExtractorVisionTransformer(nn.Module):
    """Custom ViT that returns features instead of classification output"""
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        dropout=0.0,
        attention_dropout=0.0,
    ):
        super().__init__()

        # Create a standard torchvision ViT but with a dummy classifier
        self.vit = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=1000  # Use a dummy value for initialization
        )
        
        self.vit.encoder.gradient_checkpointing = True
        
        # Remove the head, we only need the features
        self.vit.heads = nn.Identity()
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the weights properly for DINO training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use truncated normal initialization for linear layers
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Get features using the standard ViT forward pass
        # but without the classification head
        features = self.vit(x)
        
        return features

# Loss function for DINO

class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super(DINOLoss, self).__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        print(f"DINOLoss initialized with out_dim={out_dim}, teacher_temp={teacher_temp}, student_temp={student_temp}")
        
    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        epsilon = 1e-5
        
        # Process student output
        if isinstance(student_output, list):
            student_out = student_output[0] / self.student_temp
        else:
            student_out = student_output / self.student_temp
            
        # Process teacher output
        if isinstance(teacher_output, list):
            teacher_out = teacher_output[0] / self.teacher_temp
        else:
            teacher_out = teacher_output / self.teacher_temp
            
        # Center the teacher output
        teacher_out = teacher_out - self.center
        
        # Softmax for teacher and student
        # Add small epsilon to prevent exactly zero probabilities
        teacher_soft = F.softmax(teacher_out.detach(), dim=-1) + epsilon
        
        if teacher_soft.shape[0] != student_out.shape[0]:
            # If student batch is larger (e.g., includes more crops)
            if student_out.shape[0] > teacher_soft.shape[0]:
                # Calculate how many times to repeat the teacher output
                repeat_factor = student_out.shape[0] // teacher_soft.shape[0]
                teacher_soft = teacher_soft.repeat_interleave(repeat_factor, dim=0)
                
                # Handle remainder if necessary
                if teacher_soft.shape[0] < student_out.shape[0]:
                    remainder = student_out.shape[0] - teacher_soft.shape[0]
                    teacher_soft = torch.cat([teacher_soft, teacher_soft[:remainder]], dim=0)
            else:
                # If teacher batch is larger, truncate teacher output
                teacher_soft = teacher_soft[:student_out.shape[0]]
        
        
        student_log_soft = F.log_softmax(student_out, dim=-1)
        
        # Compute cross-entropy
        loss = -torch.sum(teacher_soft * student_log_soft, dim=-1).mean()
        
        # Print debug info
        print(f"Loss calculation: {loss.item()}, min teacher prob: {teacher_soft.min().item()}, max teacher prob: {teacher_soft.max().item()}")
        
        # Update center with momentum
        with torch.no_grad():
            new_center = teacher_soft[:teacher_out.shape[0]].mean(dim=0, keepdim=True)
            self.center = self.center * self.center_momentum + new_center * (1 - self.center_momentum)
                         
        # Safety check - ensure loss is not zero
        if loss.item() < 1e-8:
            print("Warning: Loss is close to zero, forcing minimum value")
            loss = torch.ones_like(loss) * 0.1
            
        return loss
    
# DINO Training function
def train_dino(
    train_h5_path,
    output_dir='./output',
    batch_size=32,
    num_epochs=100,
    lr=0.0005,
    weight_decay=0.04,
    warmup_epochs=10,
    min_lr=1e-6,
    patch_size=224,
    vit_patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
    out_dim=65536,  # dimensionality of output for DINO
    teacher_temp=0.04,
    student_temp=0.1,
    momentum_teacher=0.996,
    num_workers=4,
    device=None
):
    """
    Train a Vision Transformer with DINO on histopathology data
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    logger.info(f"Training with batch size: {batch_size} for {num_epochs} epochs")
    
    # Create student and teacher backbones (ViT)
    student_backbone = FeatureExtractorVisionTransformer(
        image_size=patch_size, 
        patch_size=vit_patch_size,
        num_layers=depth,
        num_heads=num_heads,
        hidden_dim=embed_dim,
        mlp_dim=embed_dim * 4
    )
    
    teacher_backbone = FeatureExtractorVisionTransformer(
        image_size=patch_size, 
        patch_size=vit_patch_size,
        num_layers=depth,
        num_heads=num_heads,
        hidden_dim=embed_dim,
        mlp_dim=embed_dim * 4
    )
    
    # Create DINO projection heads
    student_head = DINOHead(
        in_dim=embed_dim,
        out_dim=out_dim,
        hidden_dim=embed_dim,
        bottleneck_dim=256,
        use_bn=True
    )
    
    teacher_head = DINOHead(
        in_dim=embed_dim,
        out_dim=out_dim,
        hidden_dim=embed_dim,
        bottleneck_dim=256,
        use_bn=True
    )
    
    # Wrap models
    student = MultiCropWrapper(student_backbone, student_head)
    teacher = MultiCropWrapper(teacher_backbone, teacher_head)
    
    # Move to device
    student = student.to(device)
    teacher = teacher.to(device)
    
    # Teacher doesn't need gradients
    for p in teacher.parameters():
        p.requires_grad = False
        
    # Initialize teacher with student weights
    teacher.load_state_dict(student.state_dict())
    
    # Create loss function
    dino_loss = DINOLoss(
        out_dim=out_dim,
        teacher_temp=teacher_temp,
        student_temp=student_temp
    ).to(device)
    
    # Prepare optimizer
    param_groups = [
        {'params': [p for p in student.backbone.parameters() if p.requires_grad]},
        {'params': [p for p in student.head.parameters() if p.requires_grad]}
    ]
    
    optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    scaler = GradScaler()

    
    # Get transforms for DINO (modified for histopathology)
    # We prepare different crops: global_crops and local_crops
    from dino_h5 import get_dino_transforms
    
    global_crops_transform, local_crops_transform = get_dino_transforms(
        global_crops_size=patch_size,
        n_local_crops=2       # Number of local crops
    )
    
    # We'll create datasets with global and local transforms
    from dino_h5 import DINOMultiCropDataset
    
    dataset = DINOMultiCropDataset(
        h5_path=train_h5_path,
        global_transform=global_crops_transform,
        local_transform=local_crops_transform,
        n_global_crops=2,
        n_local_crops=2
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Scheduler with warmup and cosine decay
    warmup_steps = len(loader) * warmup_epochs
    total_steps = len(loader) * num_epochs
    
    def lr_schedule(step):
        """
        Cosine schedule with warmup
        """
        if step < warmup_steps:
            return step / warmup_steps
        else:
            decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * decay_ratio))
            return min_lr / lr + (1 - min_lr / lr) * cosine_decay
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    scaler = GradScaler()

    # Initialize best loss
    best_loss = float('inf')
    
   # Training loop
    for epoch in range(num_epochs):
        student.train()
        teacher.eval()  # Teacher always in eval mode
        running_loss = 0.0
        
        for batch_idx, (all_crops, _) in enumerate(loader):
            print(f"Batch {batch_idx}: Number of crops: {len(all_crops)}, Shapes: {[crop.shape for crop in all_crops]}")
    
            # Move to device
            all_crops = [crop.to(device) for crop in all_crops]
            
            # Print type information 
            print(f"Global crop types and devices: {[type(crop) for crop in all_crops]}, {[crop.device for crop in all_crops]}")
            
            with torch.no_grad():
                teacher_output = teacher(all_crops[:1])  # Use only first crop
                print(f"Teacher output shape: {teacher_output[0].shape if isinstance(teacher_output, list) else teacher_output.shape}")
            
            # Forward pass for student (all crops)
            student_output = student(all_crops)
            print(f"Student output shape: {student_output[0].shape if isinstance(student_output, list) else student_output.shape}")
            
            # Calculate DINO loss
            loss = dino_loss(student_output, teacher_output)
            print(f"Calculated loss: {loss.item()}")
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0)
            
            optimizer.step()
            
            # Track gradient norms
            total_grad_norm = 0.0
            param_count = 0
            for p in student.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
                    param_count += 1
            total_grad_norm = np.sqrt(total_grad_norm)
            print(f"Gradient norm: {total_grad_norm}")
            
            # Update the teacher weights with EMA
            with torch.no_grad():
                m = momentum_teacher  # momentum for EMA update
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    if param_q.requires_grad:
                        param_k.data = param_k.data * m + param_q.data * (1. - m)
            
            # Update learning rate
            scheduler.step()
            
            # Update running loss
            running_loss += loss.item()
            
            # Print progress
            if batch_idx % 10 == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(loader)}, "
                    f"Loss: {loss.item():.4f}, LR: {lr:.6f}"
               )
        
        # Calculate epoch average loss
        epoch_loss = running_loss / len(loader)
        
        # Log epoch results
        logger.info(
            f"Epoch {epoch} completed, Loss: {epoch_loss:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'student_state_dict': student.state_dict(),
                'teacher_state_dict': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(output_dir, 'dino_model_best.pt'))
    
    logger.info("Training completed!")
    return student, teacher

# Validation function for DINO
def validate_dino(
    teacher,
    val_h5_path,
    batch_size=32,
    device=None,
    num_workers=4
):
    """
    Validate a trained DINO model on histopathology data
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Validating with batch size: {batch_size}")
    
    teacher.eval()  # Set to evaluation mode
    
    # Create transforms for validation (only global crops)
    from dino_h5 import get_dino_transforms
    
    global_crops_transform, _ = get_dino_transforms(
        global_crops_size=224,  # Use same size as training
        n_local_crops=0        # No local crops for validation
    )
    
    # Create validation dataset
    from dino_h5 import DINOMultiCropDataset
    
    val_dataset = DINOMultiCropDataset(
        h5_path=val_h5_path,
        global_transform=global_crops_transform,
        local_transform=None,
        n_global_crops=2,
        n_local_crops=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Create DINO loss for validation
    dino_loss = DINOLoss(
        out_dim=teacher.head.last_layer.out_features,
        teacher_temp=0.04,
        student_temp=0.1
    ).to(device)
    
    # Run validation
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, (all_crops, _) in enumerate(val_loader):
            # Move to device
            all_crops = [crop.to(device) for crop in all_crops]
            
            # For validation, use the teacher model for both views
            # This measures consistency between different views
            teacher_output1 = teacher(all_crops[:1])
            teacher_output2 = teacher(all_crops[1:2])
            
            # Calculate consistency loss between two views
            loss = dino_loss(teacher_output2, teacher_output1)
            
            # Update total loss
            total_loss += loss.item()
            
            # Print progress
            if batch_idx % 10 == 0:
                logger.info(f"Validation Batch {batch_idx}/{len(val_loader)}, Loss: {loss.item():.4f}")
    
    # Calculate average validation loss
    val_loss = total_loss / len(val_loader)
    logger.info(f"Validation completed, Loss: {val_loss:.4f}")
    
    return val_loss

def main():
    """Main function to run training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DINO with Vision Transformer')
    parser.add_argument('--train_h5_path', type=str, required=True,
                        help='Path to training H5 file')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for saving models')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Base learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0.04,
                        help='Weight decay')
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
    parser.add_argument('--out_dim', type=int, default=65536,
                        help='Output dimension for DINO')
    parser.add_argument('--teacher_temp', type=float, default=0.04,
                        help='Teacher temperature')
    parser.add_argument('--student_temp', type=float, default=0.1,
                        help='Student temperature')
    parser.add_argument('--momentum_teacher', type=float, default=0.996,
                        help='Momentum for teacher update')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Train model
    train_dino(
        train_h5_path=args.train_h5_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        patch_size=args.patch_size,
        vit_patch_size=args.vit_patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        out_dim=args.out_dim,
        teacher_temp=args.teacher_temp,
        student_temp=args.student_temp,
        momentum_teacher=args.momentum_teacher,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main()
