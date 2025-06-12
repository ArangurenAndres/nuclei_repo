import os
import random
from src.utils.image_processing import load_image
from src.utils.visualizer import visualize_batch
import numpy as np
from PIL import Image

# --- Configuration (usually from configs/data_nuclei.yaml) ---
# For demonstration, hardcode paths relative to this script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
processed_images_path = os.path.join(project_root, 'data', 'processed', 'train_images')
processed_masks_path = os.path.join(project_root, 'data', 'processed', 'train_masks')
results_viz_path = os.path.join(project_root, 'results', 'visualizations')

# --- Main Visualization Logic ---
def run_data_visualization(num_samples: int = 4, save_output: bool = True):
    """
    Loads a few processed image-mask pairs and visualizes them.
    """
    image_files = [f for f in os.listdir(processed_images_path) if f.endswith('.png')]
    if not image_files:
        print(f"No processed images found in {processed_images_path}. Please run preprocessing first.")
        return

    # Select random image IDs for display
    selected_image_filenames = random.sample(image_files, min(num_samples, len(image_files)))

    images_to_display = []
    masks_to_display = []

    print(f"Loading {len(selected_image_filenames)} random samples for visualization...")
    for filename in selected_image_filenames:
        img_id = os.path.splitext(filename)[0] # Extract ID without extension
        img_path = os.path.join(processed_images_path, filename)
        mask_path = os.path.join(processed_masks_path, f"{img_id}_mask.png") # Our combined mask name

        img = load_image(img_path)
        mask = load_image(mask_path) # Load the *already combined* mask

        if img is not None and mask is not None:
            images_to_display.append(img)
            masks_to_display.append(mask)

    if not images_to_display:
        print("Failed to load any images for visualization.")
        return

    print("Displaying visualization...")
    save_file = os.path.join(results_viz_path, f"sample_data_batch_{num_samples}.png") if save_output else None
    visualize_batch(images_to_display, masks_to_display,
                    # image_ids=selected_image_ids, # Removed this parameter
                    num_display=num_samples,
                    save_path=save_file,
                    title_prefix="Sample Nuclei Images and Combined Masks")

if __name__ == "__main__":
    run_data_visualization(num_samples=4, save_output=True)
    print("Visualization script finished.")