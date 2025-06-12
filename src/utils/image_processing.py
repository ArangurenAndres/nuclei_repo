import os
import numpy as np
from PIL import Image
from typing import List, Tuple, Union

def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from the given path. Converts to grayscale if it's an RGB image.
    Returns: NumPy array (H, W) or (H, W, C) for color images if not converted to grayscale.
    """
    try:
        img = Image.open(image_path)
        # Convert to grayscale if it's an RGB image for consistent input to U-Net
        # The dataset description mentions varied modalities, so some might be RGB.
        if img.mode == 'RGB':
            img = img.convert('L') # Convert to grayscale
        elif img.mode != 'L': # Ensure it's grayscale if not already
             img = img.convert('L')
        return np.array(img)
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def combine_masks(mask_dir: str, img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Combines individual binary nucleus masks from a directory into a single
    binary mask for the entire image.

    Args:
        mask_dir (str): Path to the directory containing individual mask files.
        img_shape (Tuple[int, int]): The (height, width) of the corresponding image.

    Returns:
        np.ndarray: A 2D NumPy array (H, W) representing the combined binary mask,
                    where 1 indicates nucleus pixels and 0 indicates background.
    """
    combined_mask = np.zeros(img_shape, dtype=np.uint8)

    if not os.path.isdir(mask_dir):
        print(f"Warning: Mask directory not found: {mask_dir}. Returning empty mask.")
        return combined_mask

    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    if not mask_files:
        print(f"Warning: No mask files found in {mask_dir}. Returning empty mask.")
        return combined_mask

    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        try:
            # Individual masks are typically 255 for nucleus, 0 for background
            individual_mask = np.array(Image.open(mask_path).convert('L'))
            # Ensure the individual mask has the same shape as the target
            if individual_mask.shape != img_shape:
                # This should ideally not happen if data is consistent, but good to check
                print(f"Warning: Mask {mask_file} has shape {individual_mask.shape}, expected {img_shape}. Skipping.")
                continue
            # Use np.maximum to combine masks without summing up overlapping pixels
            # (though the dataset states no overlaps for individual masks)
            # This ensures pixels are 0 or 1.
            combined_mask = np.maximum(combined_mask, individual_mask / 255) # Scale to 0 or 1
        except Exception as e:
            print(f"Error processing mask {mask_file}: {e}")
            continue

    return combined_mask.astype(np.uint8)

def save_image_array(image_array: np.ndarray, save_path: str):
    """
    Saves a NumPy array image to a specified path.
    """
    try:
        Image.fromarray(image_array).save(save_path)
    except Exception as e:
        print(f"Error saving image to {save_path}: {e}")

def save_combined_mask(mask: np.ndarray, save_path: str):
    """
    Saves a combined binary mask to a specified path.
    """
    try:
        # Convert 0/1 mask back to 0/255 for standard image saving
        Image.fromarray(mask * 255).save(save_path)
    except Exception as e:
        print(f"Error saving mask to {save_path}: {e}")

def preprocess_and_save_train_data(raw_train_path: str, processed_masks_path: str, processed_images_path: str):
    """
    Iterates through the raw training data, combines masks, and saves
    the processed images and combined masks to the specified processed paths.
    """
    print(f"Starting training data preprocessing: Combining masks and saving images...")
    os.makedirs(processed_masks_path, exist_ok=True)
    os.makedirs(processed_images_path, exist_ok=True)

    image_ids = next(os.walk(raw_train_path))[1] # Get immediate subdirectories (image IDs)

    for i, img_id in enumerate(image_ids):
        print(f"Processing training image {i+1}/{len(image_ids)}: {img_id}")
        img_folder_path = os.path.join(raw_train_path, img_id)
        image_file_path = os.path.join(img_folder_path, 'images', img_id + '.png')
        mask_dir_path = os.path.join(img_folder_path, 'masks')

        img = load_image(image_file_path)
        if img is None:
            continue

        combined_mask = combine_masks(mask_dir_path, img.shape[:2]) # Ensure only H, W for mask shape

        # Save processed image and mask
        save_image_path = os.path.join(processed_images_path, f"{img_id}.png")
        save_mask_path = os.path.join(processed_masks_path, f"{img_id}_mask.png")

        save_image_array(img, save_image_path) # Save original image to processed folder
        save_combined_mask(combined_mask, save_mask_path)
    print("Training data preprocessing complete.")

def preprocess_and_save_test_images(raw_test_path: str, processed_test_images_path: str):
    """
    Iterates through the raw test data and saves the processed images
    to the specified processed path. Test data does not have masks.
    """
    print(f"Starting test data preprocessing: Saving images...")
    os.makedirs(processed_test_images_path, exist_ok=True)

    image_ids = next(os.walk(raw_test_path))[1] # Get immediate subdirectories (image IDs)

    for i, img_id in enumerate(image_ids):
        print(f"Processing test image {i+1}/{len(image_ids)}: {img_id}")
        img_folder_path = os.path.join(raw_test_path, img_id)
        image_file_path = os.path.join(img_folder_path, 'images', img_id + '.png')

        img = load_image(image_file_path)
        if img is None:
            continue

        # Save processed image
        save_image_path = os.path.join(processed_test_images_path, f"{img_id}.png")
        save_image_array(img, save_image_path)
    print("Test data preprocessing complete.")


if __name__ == "__main__":
    # Base path to your project
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Paths for training data
    #RAW_TRAIN_PATH = os.path.join(project_root, 'data', 'raw', 'stage1_train')
    #PROCESSED_TRAIN_IMAGES_PATH = os.path.join(project_root, 'data', 'processed', 'train_images')
    #PROCESSED_TRAIN_MASKS_PATH = os.path.join(project_root, 'data', 'processed', 'train_masks')

    #print("\n--- Processing Training Data ---")
    #preprocess_and_save_train_data(RAW_TRAIN_PATH, PROCESSED_TRAIN_MASKS_PATH, PROCESSED_TRAIN_IMAGES_PATH)
    #print("Training data preprocessing finished.")

    # Paths for test data (NEW)
    RAW_TEST_PATH = os.path.join(project_root, 'data', 'raw', 'stage1_test')
    PROCESSED_TEST_IMAGES_PATH = os.path.join(project_root, 'data', 'processed', 'test_images') # New folder for processed test images

    print("\n--- Processing Test Data ---")
    preprocess_and_save_test_images(RAW_TEST_PATH, PROCESSED_TEST_IMAGES_PATH)
    print("Test data preprocessing finished.")

    print("\nAll preprocessing tasks complete!")