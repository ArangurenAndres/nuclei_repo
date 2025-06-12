import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader # Added for the __main__ block

class TestDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir (str): Root directory of the test dataset (e.g., 'path/to/stage1_test').
                            Assumes structure like: root_dir/image_id/images/image_id.png
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_ids = [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
        self.image_ids.sort() # Ensure consistent order

        if not self.image_ids:
            raise RuntimeError(f"No image IDs found in {root_dir}. Please check your test dataset structure.")

        print(f"Found {len(self.image_ids)} test images in {root_dir}.")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        # This path assumes the raw stage1_test structure
        image_path = os.path.join(self.root_dir, image_id, 'images', f'{image_id}.png')

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert('L') # Convert to grayscale

        sample = {'image': image, 'image_id': image_id}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == "__main__":
    # Example usage for testing
    # Assume a dummy stage1_test structure exists:
    # nuclei_segmentation/
    # └── test_stage1_test_data/
    #     ├── image_1/
    #     │   └── images/
    #     │       └── image_1.png
    #     └── image_2/
    #         └── images/
    #             └── image_2.png
    # Create dummy files for testing purposes
    dummy_root = "test_stage1_test_data"
    os.makedirs(os.path.join(dummy_root, "image_1", "images"), exist_ok=True)
    os.makedirs(os.path.join(dummy_root, "image_2", "images"), exist_ok=True)
    Image.new('L', (256, 256), color=100).save(os.path.join(dummy_root, "image_1", "images", "image_1.png"))
    Image.new('L', (512, 512), color=200).save(os.path.join(dummy_root, "image_2", "images", "image_2.png"))

    class InferenceTransform:
        def __call__(self, sample):
            image = sample['image']
            image_id = sample['image_id']
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]) # Example normalization
            ])
            sample['image'] = transform(image)
            return sample

    print("Testing TestDataset...")
    test_dataset = TestDataset(root_dir=dummy_root, transform=InferenceTransform())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for i, batch in enumerate(test_loader):
        print(f"Batch {i+1}:")
        print(f"  Image ID: {batch['image_id'][0]}")
        print(f"  Image Tensor Shape: {batch['image'].shape}")
        # Expecting N, C, H, W -> 1, 1, H, W

    # Clean up dummy data
    import shutil
    shutil.rmtree(dummy_root)
    print("TestDataset test finished.")