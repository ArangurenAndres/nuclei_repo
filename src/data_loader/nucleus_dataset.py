# nuclei_segmentation/src/data_loader/nucleus_dataset.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any, Union
import torchvision.transforms as T
import random
import torchvision.transforms.functional as TF

from src.data_loader.transforms import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from src.utils.image_processing import load_image
from src.utils.visualizer import visualize_batch # Corrected path confirmed

# Add these DEBUG lines at the very top for diagnostics
#print(f"DEBUG: Current working directory: {os.getcwd()}")
#print(f"DEBUG: Python sys.path: {sys.path}")


class NucleusDataset(Dataset):
    """
    Dataset class for Nuclei Segmentation, handling image and mask loading,
    and patch extraction.
    """
    def __init__(self, image_dir: str, mask_dir: str, patch_size: Tuple[int, int],
                 patch_stride: Tuple[int, int], transform=None):
        """
        Args:
            image_dir (str): Directory with all the processed images.
            mask_dir (str): Directory with all the processed combined masks.
            patch_size (Tuple[int, int]): (height, width) of the patches to extract.
            patch_stride (Tuple[int, int]): (vertical stride, horizontal stride) for patch extraction.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.transform = transform
        print(self.image_dir)

        self.image_ids = [f.split('.')[0] for f in os.listdir(image_dir) if f.endswith('.png')]
        self.image_ids.sort()

        self.patch_coordinates = self._generate_patch_coordinates()
        #print(f"Initialized NucleusDataset with {len(self.image_ids)} images and {len(self.patch_coordinates)} patches.")

    def _generate_patch_coordinates(self) -> List[Dict[str, Any]]:
        """
        Generates coordinates for all possible patches across all images.
        Each coordinate entry stores: {'image_id', 'patch_x', 'patch_y'}
        """
        all_patch_coords = []
        for img_id in self.image_ids:
            img_path = os.path.join(self.image_dir, f"{img_id}.png")
            img = load_image(img_path)
            if img is None:
                print(f"Could not load image {img_id} to determine dimensions. Skipping.")
                continue

            img_h, img_w = img.shape[:2]
            patch_h, patch_w = self.patch_size
            stride_h, stride_w = self.patch_stride

            for y in range(0, img_h - patch_h + 1, stride_h):
                for x in range(0, img_w - patch_w + 1, stride_w):
                    all_patch_coords.append({
                        'image_id': img_id,
                        'patch_y_start': y,
                        'patch_x_start': x,
                        'patch_y_end': y + patch_h,
                        'patch_x_end': x + patch_w
                    })
        return all_patch_coords

    def __len__(self) -> int:
        """
        Returns the total number of patches across all images.
        """
        return len(self.patch_coordinates)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Loads an image and its mask, extracts a specific patch, and applies transforms.
        """
        coord_info = self.patch_coordinates[idx]
        img_id = coord_info['image_id']
        y_start, x_start = coord_info['patch_y_start'], coord_info['patch_x_start']
        y_end, x_end = coord_info['patch_y_end'], coord_info['patch_x_end']

        # Construct file paths
        image_path = os.path.join(self.image_dir, f"{img_id}.png")
        mask_path = os.path.join(self.mask_dir, f"{img_id}_mask.png")

        # Load full image and mask
        image = load_image(image_path)
        mask = load_image(mask_path) # Mask loaded here, should be fine (0-255 uint8)

        if image is None or mask is None:
            raise RuntimeError(f"Failed to load image/mask for ID: {img_id}")

        # Extract the patch
        image_patch = image[y_start:y_end, x_start:x_end]
        mask_patch = mask[y_start:y_end, x_start:x_end] # Patch extracted here

        # --- DEBUG PRINTS FOR MASK PATCH ---
        #print(f"DEBUG (getitem): Mask Patch for {img_id} [{y_start}:{y_end}, {x_start}:{x_end}]")
        #print(f"DEBUG (getitem):   Shape: {mask_patch.shape}, Dtype: {mask_patch.dtype}")
        #print(f"DEBUG (getitem):   Min value: {mask_patch.min()}, Max value: {mask_patch.max()}, Sum of pixels: {mask_patch.sum()}")
        # --- END DEBUG PRINTS ---

        sample = {'image': image_patch, 'mask': mask_patch}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Example usage (for testing the dataset directly)
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    PROCESSED_IMAGES_PATH = os.path.join(project_root, 'data', 'processed', 'train_images')
    PROCESSED_MASKS_PATH = os.path.join(project_root, 'data', 'processed', 'train_masks')

    PATCH_SIZE = (256, 256)
    PATCH_STRIDE = (128, 128)

    data_transforms = T.Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(degrees=(0, 20)),
        ToTensor() # This now correctly receives NumPy arrays
    ])


    print("\nCreating NucleusDataset...")
    dataset = NucleusDataset(
        image_dir=PROCESSED_IMAGES_PATH,
        mask_dir=PROCESSED_MASKS_PATH,
        patch_size=PATCH_SIZE,
        patch_stride=PATCH_STRIDE,
        transform=data_transforms
    )

    BATCH_SIZE = 4
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print(f"\nFetching a batch from DataLoader (Batch Size: {BATCH_SIZE})...")
    for i, batch in enumerate(dataloader):
        images_batch = batch['image']
        masks_batch = batch['mask']

        print(f"Batch {i+1}:")
        print(f"  Images batch shape: {images_batch.shape}")
        print(f"  Masks batch shape: {masks_batch.shape}")

        images_np = [img.squeeze().cpu().numpy() * 255 for img in images_batch]
        masks_np = [mask.squeeze().cpu().numpy() * 255 for mask in masks_batch]

        visualize_batch(images_np, masks_np, num_display=BATCH_SIZE, title_prefix=f"Sample Patches from DataLoader (Batch {i+1})")

        if i == 0:
            break

    print("\nDataLoader demonstration complete.")