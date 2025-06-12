import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os # To list directory contents and join paths

def visualize_segmentation(image_path: str, mask_path: str):
    """
    Displays an original image, its binary mask, and an overlay of the two.

    Args:
        image_path (str): Path to the original image file (e.g., .png, .jpg).
        mask_path (str): Path to the corresponding mask file (e.g., .png).
                         Assumes the mask is binary (0 for background, >0 for foreground).
    """
    print(f"Loading image: {image_path}")
    print(f"Loading mask: {mask_path}")

    try:
        # Load the image. We convert to 'L' (grayscale) as typical for
        # microscopy images and for consistency with masks.
        image = np.array(Image.open(image_path).convert('L'))
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}. Please double-check the path.")
        return
    except Exception as e:
        print(f"Error loading image from {image_path}: {e}")
        return

    try:
        # Load the mask. Convert to 'L' (grayscale) to ensure consistent 0/255 values.
        mask = np.array(Image.open(mask_path).convert('L'))
        # Convert mask to binary (0 and 1). Assuming nuclei are represented by
        # non-zero values (e.g., 255) and background by 0.
        mask_binary = (mask > 0).astype(np.uint8)
    except FileNotFoundError:
        print(f"Error: Mask file not found at {mask_path}. Please check the mask path and ensure it corresponds to the image.")
        return
    except Exception as e:
        print(f"Error loading mask from {mask_path}: {e}")
        return

    # Basic check for matching dimensions
    if image.shape != mask_binary.shape:
        print(f"Error: Image and mask dimensions do not match. Image: {image.shape}, Mask: {mask_binary.shape}")
        return

    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5)) # 1 row, 3 columns, figure size 15x5 inches

    # --- Subplot 1: Original Image ---
    axes[0].imshow(image, cmap='gray') # Display the grayscale image
    axes[0].set_title('Original Image')
    axes[0].axis('off') # Hide axes ticks and labels for cleaner image display

    # --- Subplot 2: Ground Truth Mask ---
    # Use 'gray' colormap for the binary mask (0=black, 1=white)
    axes[1].imshow(mask_binary, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    # --- Subplot 3: Image + Mask Overlay ---
    axes[2].imshow(image, cmap='gray') # Start by displaying the original image
    # Overlay the binary mask with transparency.
    # The 'Reds' colormap will make the masked regions (where mask_binary is 1)
    # appear in a semi-transparent red color.
    # Adjust 'alpha' (0.0 for fully transparent, 1.0 for fully opaque) as desired.
    axes[2].imshow(mask_binary, cmap='Reds', alpha=0.4) # Overlay with 40% opacity
    axes[2].set_title('Image + Mask Overlay')
    axes[2].axis('off')

    plt.tight_layout() # Adjust subplot parameters for a tight layout
    plt.show() # Display the plot

# Example Usage (This part will run if you execute the script directly)
if __name__ == '__main__':
    print("--- Attempting to visualize real data from data/processed ---")

    # Define the directories for images and masks relative to the project root
    # IMPORTANT: Ensure you run this script from your project's root directory
    # (the one containing 'data/', 'src/', 'config.yaml', etc.)
    train_images_dir = 'data/processed/train_images'
    train_masks_dir = 'data/processed/train_masks'

    # Check if directories exist
    if not os.path.isdir(train_images_dir):
        print(f"Error: Image directory not found: '{train_images_dir}'")
        print("Please ensure you are running this script from the project's root directory (where 'data' is located).")
        exit()
    if not os.path.isdir(train_masks_dir):
        print(f"Error: Mask directory not found: '{train_masks_dir}'")
        print("Please ensure you are running this script from the project's root directory (where 'data' is located).")
        exit()

    # Find the first .png image in the train_images directory to use as an example
    example_image_name = None
    for filename in os.listdir(train_images_dir):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            example_image_name = filename
            break

    if example_image_name is None:
        print(f"Error: No image files (.png, .jpg, .jpeg) found in '{train_images_dir}'.")
        print("Please ensure your 'data/processed/train_images' folder contains images.")
        exit()

    image_path = os.path.join(train_images_dir, example_image_name)

    # --- UPDATED LOGIC FOR MASK PATH WITH '_mask' SUFFIX ---
    # This logic assumes: if your image is 'image_id.png', its mask is 'image_id_mask.png'
    # in the masks directory.
    base_name, ext = os.path.splitext(example_image_name) # Splits 'filename.png' into ('filename', '.png')
    mask_name = f"{base_name}_mask{ext}" # Creates 'filename_mask.png'
    mask_path = os.path.join(train_masks_dir, mask_name)

    # Call the visualization function
    visualize_segmentation(image_path, mask_path)

    print("\n--- Real data visualization complete ---")
    print("Tip: If you want to visualize a specific image, you can manually change 'example_image_name'")
    print("     to the exact filename (e.g., `example_image_name = 'specific_image_id.png'`)")
    print("     and ensure its corresponding mask exists in the mask directory.")