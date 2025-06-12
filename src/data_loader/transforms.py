# nuclei_segmentation/src/data_loader/transforms.py

import torch
import torchvision.transforms.functional as TF
import random
from PIL import Image
import numpy as np

class ToTensor:
    """Convert numpy arrays (H, W) to torch tensors (C, H, W)."""
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        # Add channel dimension (C=1 for grayscale)
        image = torch.from_numpy(image).float().unsqueeze(0) / 255.0 # Normalize to [0, 1]
        mask = torch.from_numpy(mask).float().unsqueeze(0) # Mask is already 0/1 or 0/255 and ToTensor makes it float32. Here we want 0/1 for masks.
        return {'image': image, 'mask': mask}

class RandomHorizontalFlip:
    """Horizontally flips the given image and mask randomly with a given probability."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if random.random() < self.p:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        return {'image': image, 'mask': mask}

class RandomVerticalFlip:
    """Vertically flips the given image and mask randomly with a given probability."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if random.random() < self.p:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        return {'image': image, 'mask': mask}

class RandomRotation:
    """Rotates the image and mask by a random angle."""
    def __init__(self, degrees=(0, 180)):
        self.degrees = degrees

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        angle = random.uniform(self.degrees[0], self.degrees[1])

        image_pil = Image.fromarray(image)
        # Masks are 0/255 (uint8) coming in, convert to PIL. The PIL image will maintain 0/255 values
        mask_pil = Image.fromarray(mask)

        image_rotated_pil = TF.rotate(image_pil, angle, interpolation=Image.BICUBIC)
        mask_rotated_pil = TF.rotate(mask_pil, angle, interpolation=Image.NEAREST)

        # Convert back to numpy arrays
        image_rotated = np.array(image_rotated_pil)

        
        mask_rotated = (np.array(mask_rotated_pil) / 255.0).astype(np.float32)

        

        return {'image': image_rotated, 'mask': mask_rotated}