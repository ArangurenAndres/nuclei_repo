import torch
import yaml
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms as T

# Add the project root to the sys.path to allow importing modules from src/
# This assumes the script is in a subdirectory like src/inference/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(project_root)

# Import custom modules
from src.models.unet import UNET
# from src.data_loader.test_dataset import TestDataset # Not directly used for processed images

# Helper function to get device
def get_device(device_str: str):
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "mps" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")

def load_model(model_path: str, config: dict, device: torch.device):
    """Loads a pre-trained UNET model."""
    model_config = config['model']['params']
    model = UNET(
        in_channels=model_config['in_channels'],
        out_channels=model_config['out_channels'],
        features=model_config['features']
    ).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Please train the model first.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set to evaluation mode
    print(f"Model loaded from {model_path}")
    return model

def create_inference_transforms():
    """
    Creates transforms for inference.
    Assumes images are loaded as PIL Image 'L' (grayscale) and scaled to [0, 1] then normalized.
    """
    return T.Compose([
        T.ToTensor(),
        # Normalize with the same mean/std used during training
        # Assuming 0.5 mean and 0.5 std for images scaled to [-1, 1] range after ToTensor
        T.Normalize(mean=[0.5], std=[0.5])
    ])

def predict_full_image(model: torch.nn.Module, full_image_tensor: torch.Tensor, patch_size: tuple, patch_stride: tuple, device: torch.device):
    """
    Predicts a mask for a full-sized image by patching, predicting, and re-stitching.

    Args:
        model (nn.Module): The trained segmentation model.
        full_image_tensor (torch.Tensor): The full input image tensor (C, H, W).
        patch_size (tuple): (height, width) of patches.
        patch_stride (tuple): (height, width) stride for overlapping patches.
        device (torch.device): Device to perform inference on.

    Returns:
        tuple: (np.ndarray, np.ndarray) A tuple containing:
               - The predicted full-resolution binary mask (H, W).
               - The full-resolution probability map (H, W) before binarization.
    """
    _, H, W = full_image_tensor.shape
    patch_H, patch_W = patch_size
    stride_H, stride_W = patch_stride

    # Pad image to handle edge cases and ensure all areas are covered by patches
    # Calculate padding needed to ensure image dimensions are multiples of stride and cover patches
    padded_H = int(np.ceil(max(H, patch_H) / stride_H)) * stride_H if H % stride_H != 0 else H
    padded_W = int(np.ceil(max(W, patch_W) / stride_W)) * stride_W if W % stride_W != 0 else W
    
    # If the image is smaller than the patch, ensure padded_H/W are at least patch_H/W
    padded_H = max(padded_H, patch_H)
    padded_W = max(padded_W, patch_W)

    pad_bottom = padded_H - H
    pad_right = padded_W - W
    
    # Pad only if necessary
    if pad_bottom > 0 or pad_right > 0:
        padded_image = torch.nn.functional.pad(full_image_tensor, (0, pad_right, 0, pad_bottom), mode='reflect')
    else:
        padded_image = full_image_tensor

    output_mask_sum = torch.zeros((1, padded_H, padded_W), dtype=torch.float32).to(device)
    overlap_counts = torch.zeros((1, padded_H, padded_W), dtype=torch.float32).to(device)

    # Iterate through patches
    for y in range(0, padded_H - patch_H + 1, stride_H):
        for x in range(0, padded_W - patch_W + 1, stride_W):
            patch = padded_image[:, y:y + patch_H, x:x + patch_W].unsqueeze(0).to(device) # Add batch dim

            with torch.no_grad():
                pred_patch_logits = model(patch)
                pred_patch_prob = torch.sigmoid(pred_patch_logits) # Get probabilities

            output_mask_sum[:, y:y + patch_H, x:x + patch_W] += pred_patch_prob.squeeze(0) # Remove batch dim
            overlap_counts[:, y:y + patch_H, x:x + patch_W] += 1 # Mark pixels as covered

    overlap_counts[overlap_counts == 0] = 1e-6 # Avoid division by zero for uncovered pixels (should be rare with overlap)

    final_mask_prob = output_mask_sum / overlap_counts
    
    final_mask_prob = final_mask_prob[:, :H, :W] # Crop back to original image dimensions

    # Convert probabilities to numpy array for return
    final_mask_prob_np = final_mask_prob.cpu().numpy().squeeze()

    # Convert probabilities to binary mask (0 or 1)
    final_mask_binary_np = (final_mask_prob_np > 0.5).astype(np.uint8)

    return final_mask_binary_np, final_mask_prob_np # Return both binary and probability map


def evaluate_model():
    # 1. Load configuration
    config_path = os.path.join(project_root, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    training_config = config['training']
    inference_config = config['inference'] # This needs to exist in config.yaml!
    logging_config = config['logging']

    # 2. Set up device
    device = get_device(training_config['device'])
    print(f"Using device: {device}")

    # 3. Load Model
    model_path = os.path.join(project_root, logging_config['checkpoint_dir'], "best_model.pth")
    model = load_model(model_path, config, device)

    # 4. Set up Test Data Loading from processed folder
    processed_test_image_dir = os.path.join(project_root, 'data', 'processed', 'test_images')
    
    # IMPORTANT NOTE:
    # The official test set for this competition (like Kaggle 2018 Data Science Bowl)
    # does NOT provide ground truth masks. Therefore, we cannot compute metrics
    # like Dice Coefficient for these images.
    # Dice Coefficient is typically computed on a held-out VALIDATION set during training.

    # Check if processed test images exist
    if not os.path.exists(processed_test_image_dir) or not os.listdir(processed_test_image_dir):
        print(f"Error: Processed test images not found in {processed_test_image_dir}.")
        print("Please run 'python src/data_loader/image_processing.py' first to prepare the test data.")
        return

    # Get list of image paths from the processed directory
    test_image_paths = []
    for filename in os.listdir(processed_test_image_dir):
        if filename.endswith('.png'): # Assuming processed images are png
            test_image_paths.append(os.path.join(processed_test_image_dir, filename))
    test_image_paths.sort() # Ensure consistent order

    if not test_image_paths:
        print(f"No processed test images found in {processed_test_image_dir}.")
        return

    # Define transforms for the test images (just ToTensor and normalization)
    inference_transforms = create_inference_transforms()

    print(f"Found {len(test_image_paths)} processed test images to evaluate.")

    # 5. Create output directory for predicted masks
    output_mask_dir = os.path.join(project_root, inference_config['output_mask_dir'])
    os.makedirs(output_mask_dir, exist_ok=True)
    print(f"Predicted masks will be saved to: {output_mask_dir}")

    # 6. Perform Inference and Plot Results
    print("\nStarting inference on test data...")
    num_plotted = 0
    display_limit = inference_config.get('display_limit', 5) # Default to 5 plots

    # Initialize a counter for the 'test_image_n' naming
    test_image_counter = 0

    # Prepare for reporting results
    report_data = []

    for i, img_path in enumerate(tqdm(test_image_paths, desc="Predicting masks")):
        original_image_id = os.path.splitext(os.path.basename(img_path))[0] # Original ID from filename
        test_image_counter += 1
        display_image_name = f"test_image_{test_image_counter}" # Short name for display (e.g., test_image_1)

        # Load image (processed image is already grayscale)
        original_image_pil = Image.open(img_path).convert('L')
        
        # Apply inference transforms (ToTensor and normalization)
        original_image_tensor = inference_transforms(original_image_pil) # (C, H, W)

        # Predict full mask using patch-based inference
        predicted_mask_binary_np, predicted_mask_prob_np = predict_full_image(
            model,
            original_image_tensor,
            tuple(training_config['patch_size']),
            tuple(training_config['patch_stride']),
            device
        )
        
        # Calculate confidence: Mean probability of foreground pixels
        if np.sum(predicted_mask_binary_np) > 0:
            confidence = predicted_mask_prob_np[predicted_mask_binary_np == 1].mean()
        else:
            confidence = 0.0 # No foreground pixels detected, confidence is 0

        # Convert original image tensor back to numpy for plotting (un-normalize)
        # Assumes the original image was normalized to [-1, 1] using (x - 0.5) / 0.5
        original_image_display = (original_image_tensor.cpu().numpy().squeeze() * 0.5 + 0.5) * 255
        original_image_display = original_image_display.astype(np.uint8)
        
        # Save the predicted mask (using the binary mask)
        predicted_mask_pil = Image.fromarray((predicted_mask_binary_np * 255).astype(np.uint8))
        mask_save_path = os.path.join(output_mask_dir, f"{original_image_id}_pred_mask.png") # Keep original ID for saving
        predicted_mask_pil.save(mask_save_path)

        # Store results for report
        report_data.append({
            'image_id': original_image_id,
            'display_name': display_image_name,
            'confidence': confidence,
            'dice_coefficient': 'N/A (No Ground Truth)' # Explicitly state no Dice
        })


        # Plot if within display limit
        if display_limit == -1 or num_plotted < display_limit:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(original_image_display, cmap='gray')
            plt.title(f"Image: {display_image_name}") # Use short name in title
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(predicted_mask_binary_np, cmap='gray')
            # Use short name and add confidence in title
            plt.title(f"Mask: {display_image_name} (Confidence: {confidence:.4f})")
            plt.axis('off')
            
            plt.suptitle(f"Inference Results for {display_image_name}")
            plt.show()
            num_plotted += 1

    print(f"\nInference complete. Predicted masks saved to {output_mask_dir}")
    if display_limit != 0:
        print(f"Displayed {num_plotted} images (up to display_limit={display_limit}).")

    # 7. Generate and print the report
    print("\n--- Inference Report ---")
    print(f"{'Image ID':<40} | {'Display Name':<15} | {'Confidence':<12} | {'Dice Coeff.':<15}")
    print("-" * 90)
    for entry in report_data:
        print(f"{entry['image_id']:<40} | {entry['display_name']:<15} | {entry['confidence']:.4f}{' ':>7} | {entry['dice_coefficient']:<15}")
    print("-" * 90)
    print("Note: Dice Coefficient for the test set cannot be computed as ground truth masks are not provided.")
    print("For model evaluation, Dice Coefficient is typically computed on a held-out validation set during training.")


if __name__ == "__main__":
    evaluate_model()