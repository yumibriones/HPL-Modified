"""
Simplified H5 Dataset Implementation for Barlow Twins SSL
"""

import os
import torch
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
import logging
from skimage import color

class RandomStainNormalization(object):
    def __init__(self, p=0.5):
        self.prob = p
        
    def __call__(self, img):
        if random.random() < self.prob:
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
        else:
            return img

logger = logging.getLogger(__name__)

class SimpleH5Dataset(Dataset):
    """Very simple dataset for loading histopathology patches from H5 files"""
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        
        # Find the image key
        with h5py.File(self.h5_path, 'r') as f:
            keys = list(f.keys())
            image_keys = ['img', 'images', 'x', 'data']
            
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

def get_barlow_twins_transforms(patch_size=224):
    """
    Create transforms for Barlow Twins SSL
    Returns two transformations to create two views
    """
    # First, add any custom transform classes
    class GaussianNoise(object):
        def __init__(self, std=0.05):
            self.std = std
            
        def __call__(self, img):
            img_array = np.array(img).astype(np.float32) / 255.0
            noise = np.random.normal(0, self.std, img_array.shape)
            noisy_img = np.clip(img_array + noise, 0, 1.0)
            return Image.fromarray((noisy_img * 255).astype(np.uint8))

    class GaussianBlur(object):
        def __init__(self, radius_min=0.1, radius_max=2.0):
            self.radius_min = radius_min
            self.radius_max = radius_max

        def __call__(self, img):
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
            
    # Make sure to add the imports at the top of the file
    # from PIL import ImageFilter
    # import random
    
    # Define augmentation for the first view
    color_jitter = transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
    )
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Standard ImageNet stats as starting point
        std=[0.229, 0.224, 0.225]
    )
    
    # Define the first augmentation pipeline
    transform1 = transforms.Compose([
        transforms.RandomResizedCrop(patch_size, scale=(0.14, 1.0)),  # Increased min area
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation([90, 90])], p=0.5),  # 90 degree rotations
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomApply([GaussianBlur(0.1, 2.0)], p=0.5),  # Add blur
        transforms.RandomApply([GaussianNoise(std=0.05)], p=0.3),  # Add noise
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([RandomStainNormalization(p=1.0)], p=0.5),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Define the second augmentation pipeline (different random seeds)
    transform2 = transforms.Compose([
        transforms.RandomResizedCrop(patch_size, scale=(0.14, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation([90, 90])], p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomApply([GaussianBlur(0.1, 2.0)], p=0.5),
        transforms.RandomApply([GaussianNoise(std=0.05)], p=0.3),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([RandomStainNormalization(p=1.0)], p=0.5),
        transforms.ToTensor(),
        normalize,
    ])
    
    return transform1, transform2
