"""
DINO Dataset Implementation for Histopathology SSL
"""

import os
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
import logging
import random
try:
    from skimage import color
except ImportError:
    logging.warning("skimage not found. Stain normalization won't be available.")

logger = logging.getLogger(__name__)

class RandomGaussianBlur(object):
    """
    Gaussian blur with random sigma between a min and a max value
    """
    def __init__(self, radius_min=0.1, radius_max=2.0):
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        radius = random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

class GaussianNoise(object):
    """
    Add gaussian noise to the image
    """
    def __init__(self, std=0.05):
        self.std = std
        
    def __call__(self, img):
        img_array = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, self.std, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 1.0)
        return Image.fromarray((noisy_img * 255).astype(np.uint8))

class RandomStainNormalization(object):
    """
    Apply random stain normalization for histopathology images
    """
    def __init__(self, p=0.5):
        self.prob = p
        
    def __call__(self, img):
        if random.random() < self.prob:
            try:
                # Convert to LAB color space
                np_img = np.array(img)
                lab = color.rgb2lab(np_img)
                
                # Randomly adjust mean and std of each channel
                for i in range(3):
                    mean_shift = random.uniform(-10, 10)
                    std_scale = random.uniform(0.8, 1.2)
                    
                    channel = lab[:,:,i]
                    mean = np.mean(channel)
                    std = np.std(channel)
                    
                    # Shift mean and scale std
                    channel = ((channel - mean) * std_scale) + mean + mean_shift
                    
                    # Clip to valid range
                    if i == 0:  # L channel
                        channel = np.clip(channel, 0, 100)
                    else:  # a,b channels
                        channel = np.clip(channel, -128, 127)
                        
                    lab[:,:,i] = channel
                    
                # Convert back to RGB
                rgb = color.lab2rgb(lab)
                return Image.fromarray((rgb * 255).astype(np.uint8))
            except (NameError, ImportError):
                # If skimage is not available, return original image
                return img
        else:
            return img

def get_dino_transforms(global_crops_size=224, n_local_crops=0):
    """
    Create transforms for DINO SSL
    Returns two transformations: one for global crops and one for local crops
    Both will output the same size images, but local crops will be from smaller regions
    """
    # Histogram normalization parameters for histopathology
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Use ImageNet stats as starting point
        std=[0.229, 0.224, 0.225]
    )
    
    # Color augmentation
    color_jitter = transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
    )
    
    # First global crop transform (larger crops)
    global_crops_transform = transforms.Compose([
        transforms.RandomResizedCrop(global_crops_size, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation([90, 90])], p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomApply([RandomGaussianBlur(0.1, 2.0)], p=0.5),
        transforms.RandomApply([GaussianNoise(std=0.05)], p=0.3),
        transforms.RandomApply([RandomStainNormalization()], p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Local crops transform (from smaller regions but same output size)
    local_crops_transform = transforms.Compose([
        # Key change: Use same output size as global crops but smaller scale
        transforms.RandomResizedCrop(global_crops_size, scale=(0.05, 0.14)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation([90, 90])], p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomApply([RandomGaussianBlur(0.1, 2.0)], p=0.5),
        transforms.RandomApply([GaussianNoise(std=0.05)], p=0.3),
        transforms.RandomApply([RandomStainNormalization()], p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    
    return global_crops_transform, local_crops_transform

class SimpleH5Dataset(Dataset):
    """Very simple dataset for loading histopathology patches from H5 files"""
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        
        # Find the image key
        with h5py.File(self.h5_path, 'r') as f:
            keys = list(f.keys())
            image_keys = ['img', 'images', 'valid_img']
            
            self.img_key = None
            for key in image_keys:
                if key in keys:
                    self.img_key = key
                    break
            
            if self.img_key is None:
                raise KeyError(f"Could not find valid image key in {keys}")
            
            # Get dataset info
            self.length = f[self.img_key].shape[0]
            self.img_shape = f[self.img_key].shape[1:]
        
        logger.info(f"Loaded H5 dataset with {self.length} samples of shape {self.img_shape}")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Load image from h5 file
        with h5py.File(self.h5_path, 'r') as f:
            img_data = f[self.img_key][idx]
        
        # Convert to PIL Image
        if img_data.dtype == np.uint8:
            image = Image.fromarray(img_data)
        else:
            # Normalize to 0-255 if not uint8
            img_data = ((img_data - img_data.min()) / (img_data.max() - img_data.min()) * 255).astype(np.uint8)
            image = Image.fromarray(img_data)
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image

class DINOMultiCropDataset(Dataset):
    """
    Dataset for DINO multi-crop strategy
    Creates multiple views of the same image:
    - 2 global crops
    - N local crops
    All crops will have the same size but come from regions of different scales
    """
    def __init__(self, h5_path, global_transform, local_transform, n_global_crops=2, n_local_crops=0):
        self.dataset = SimpleH5Dataset(h5_path)
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Load the image
        with h5py.File(self.dataset.h5_path, 'r') as f:
            img_data = f[self.dataset.img_key][idx]
        
        # Convert to PIL Image
        if img_data.dtype == np.uint8:
            image = Image.fromarray(img_data)
        else:
            # Normalize to 0-255 if not uint8
            img_data = ((img_data - img_data.min()) / (img_data.max() - img_data.min()) * 255).astype(np.uint8)
            image = Image.fromarray(img_data)
        
        # Create global crops
        global_crops = [self.global_transform(image) for _ in range(self.n_global_crops)]
        
        # Create local crops
        local_crops = [self.local_transform(image) for _ in range(self.n_local_crops)]
        
        # All crops (global + local)
        crops = global_crops + local_crops
        
        return crops, idx

