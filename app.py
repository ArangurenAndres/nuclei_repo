import os
import sys
import yaml
import torch
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, session
import numpy as np
from PIL import Image
from tqdm import tqdm # For server-side progress indication during batch prediction

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

from evaluate import get_device, load_model, create_inference_transforms, predict_full_image


app = Flask(__name__)
app.secret_key = 'your_super_secret_and_very_long_random_key_here_for_flask_session' 

# --- Global Model and Config Loading ---
# These will be loaded once when the Flask application starts.
MODEL = None
DEVICE = None
CONFIG = None
INFERENCE_TRANSFORMS = None

# Define image directories (paths relative to project_root)
PROCESSED_TRAIN_IMAGES_DIR = os.path.join(project_root, 'data', 'processed', 'train_images')
PROCESSED_TRAIN_MASKS_DIR = os.path.join(project_root, 'data', 'processed', 'train_masks')
PROCESSED_TEST_IMAGES_DIR = os.path.join(project_root, 'data', 'processed', 'test_images')
FLASK_PREDICTIONS_DIR = None # This will be set from config and created if needed

# Global lists for image IDs and current index for navigation
TRAIN_IMAGE_IDS = []
TEST_IMAGE_IDS = [] # Will be populated after evaluation
CURRENT_TRAIN_INDEX = 0
CURRENT_TEST_INDEX = 0

def init_app_globals():
    """Initializes global variables for model, config, and image paths."""
    global MODEL, DEVICE, CONFIG, INFERENCE_TRANSFORMS, FLASK_PREDICTIONS_DIR, TRAIN_IMAGE_IDS

    # Load configuration from config.yaml
    config_path = os.path.join(project_root, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, 'r') as f:
        CONFIG = yaml.safe_load(f)

    # Set up computational device (CPU, CUDA, or MPS)
    DEVICE = get_device(CONFIG['training']['device'])
    print(f"Flask app using device: {DEVICE}")

    # Load the pre-trained UNET model
    model_path = os.path.join(project_root, CONFIG['logging']['checkpoint_dir'], "best_model.pth")
    MODEL = load_model(model_path, CONFIG, DEVICE)
    INFERENCE_TRANSFORMS = create_inference_transforms()

    # Define and create the directory for Flask-generated predictions
    FLASK_PREDICTIONS_DIR = os.path.join(project_root, CONFIG['inference']['flask_output_mask_dir'])
    os.makedirs(FLASK_PREDICTIONS_DIR, exist_ok=True)
    print(f"Predicted masks for Flask app will be saved to: {FLASK_PREDICTIONS_DIR}")

    # Load image IDs from the processed training set for Browse
    TRAIN_IMAGE_IDS = sorted([f.split('.')[0] for f in os.listdir(PROCESSED_TRAIN_IMAGES_DIR) if f.endswith('.png')])
    if not TRAIN_IMAGE_IDS:
        raise RuntimeError(f"No processed train images found in {PROCESSED_TRAIN_IMAGES_DIR}. Please preprocess data first.")
    print(f"Loaded {len(TRAIN_IMAGE_IDS)} training images for Browse.")


# --- Helper Route to Serve Images ---
@app.route('/images/<folder>/<filename>')
def serve_image(folder, filename):
    """
    Serves image files from specified processed data folders or flask_predictions.
    This allows the HTML to access images dynamically.
    """
    if folder == 'train_images':
        directory = PROCESSED_TRAIN_IMAGES_DIR
    elif folder == 'train_masks':
        directory = PROCESSED_TRAIN_MASKS_DIR
    elif folder == 'flask_predictions':
        directory = FLASK_PREDICTIONS_DIR
    elif folder == 'test_images': # To serve original test images in test results view
        directory = PROCESSED_TEST_IMAGES_DIR
    else:
        return "Invalid image folder", 404

    return send_from_directory(directory, filename)


# --- Flask Routes ---

@app.route('/', methods=['GET'])
def train_view():
    """
    Displays the main page with training image-mask pairs.
    Allows navigation through training data. This is the default view.
    """
    global CURRENT_TRAIN_INDEX

    # Handle navigation (Next/Previous buttons) for training data
    action = request.args.get('action')
    if action == 'next':
        CURRENT_TRAIN_INDEX = (CURRENT_TRAIN_INDEX + 1) % len(TRAIN_IMAGE_IDS)
    elif action == 'prev':
        CURRENT_TRAIN_INDEX = (CURRENT_TRAIN_INDEX - 1 + len(TRAIN_IMAGE_IDS)) % len(TRAIN_IMAGE_IDS)

    current_image_id = TRAIN_IMAGE_IDS[CURRENT_TRAIN_INDEX]

    # URLs for the current original image and its ground truth mask
    image_url = url_for('serve_image', folder='train_images', filename=f"{current_image_id}.png")
    gt_mask_url = url_for('serve_image', folder='train_masks', filename=f"{current_image_id}_mask.png")

    # Clear any previous test results session flag or test image list
    session.pop('show_test_results', None)
    session.pop('test_image_ids', None)
    session.pop('current_test_index', None)
    
    return render_template('index.html',
                           view='train', # Indicate that we are in the training data view
                           current_image_id=current_image_id,
                           image_url=image_url,
                           mask_url=gt_mask_url, # For train view, mask_url is ground truth
                           num_images=len(TRAIN_IMAGE_IDS),
                           current_index=CURRENT_TRAIN_INDEX + 1) # +1 for 1-based indexing in UI


@app.route('/evaluate_test_set', methods=['POST'])
def evaluate_test_set():
    """
    Triggers batch evaluation of all images in the test set.
    Saves predicted masks to FLASK_PREDICTIONS_DIR.
    Then redirects to the test results view.
    """
    global MODEL, DEVICE, CONFIG, INFERENCE_TRANSFORMS, TEST_IMAGE_IDS

    if MODEL is None or INFERENCE_TRANSFORMS is None:
        return "Server error: Model or transforms not loaded. Please restart the server.", 500

    test_image_filenames = sorted([f for f in os.listdir(PROCESSED_TEST_IMAGES_DIR) if f.endswith('.png')])
    if not test_image_filenames:
        print(f"No test images found in {PROCESSED_TEST_IMAGES_DIR}. Skipping evaluation.")
        # Even if no images, set the session flags to go to test results view but with an empty list
        session['test_image_ids'] = []
        session['current_test_index'] = 0
        session['show_test_results'] = True 
        return redirect(url_for('test_results_view'))

    print(f"Starting batch inference on {len(test_image_filenames)} test images...")

    # Get patch size and stride from config
    patch_size = tuple(CONFIG['training']['patch_size'])
    patch_stride = tuple(CONFIG['training']['patch_stride'])

    # List to store IDs of images for which predictions were successfully made
    processed_test_ids = []

    for filename in tqdm(test_image_filenames, desc="Predicting masks"):
        image_id = os.path.splitext(filename)[0] # Original ID from filename
        image_path = os.path.join(PROCESSED_TEST_IMAGES_DIR, filename)

        try:
            original_image_pil = Image.open(image_path).convert('L')
            original_image_tensor = INFERENCE_TRANSFORMS(original_image_pil)
        except Exception as e:
            print(f"Error loading test image {filename}: {e}")
            continue # Skip to next image if there's a loading error

        try:
            predicted_mask_binary_np, _ = predict_full_image(
                MODEL,
                original_image_tensor,
                patch_size,
                patch_stride,
                DEVICE
            )

            # Save the predicted mask
            predicted_mask_filename = f"{image_id}_pred_mask.png"
            predicted_mask_pil = Image.fromarray((predicted_mask_binary_np * 255).astype(np.uint8))
            predicted_mask_pil.save(os.path.join(FLASK_PREDICTIONS_DIR, predicted_mask_filename))
            processed_test_ids.append(image_id) # Add to list of successfully processed IDs
        except Exception as e:
            print(f"Error predicting or saving mask for {filename}: {e}")
            continue # Skip to next image if prediction fails


    print("Batch inference complete.")

    # Populate TEST_IMAGE_IDS with the IDs for which we just made predictions
    TEST_IMAGE_IDS = sorted(processed_test_ids) # Use the IDs that were actually processed
    session['test_image_ids'] = TEST_IMAGE_IDS # Store in session to persist across redirects
    session['current_test_index'] = 0 # Reset test index
    session['show_test_results'] = True # Set flag to display test results

    return redirect(url_for('test_results_view'))


@app.route('/test_results', methods=['GET'])
def test_results_view():
    """
    Displays the test image-predicted mask pairs after evaluation.
    Allows navigation through test data. This is accessed after batch prediction.
    """
    global CURRENT_TEST_INDEX, TEST_IMAGE_IDS

    # Retrieve TEST_IMAGE_IDS from session, as it might have been populated by evaluate_test_set
    if not TEST_IMAGE_IDS and 'test_image_ids' in session:
        TEST_IMAGE_IDS = session.get('test_image_ids', [])
        CURRENT_TEST_INDEX = session.get('current_test_index', 0)
    
    # If after checking session, TEST_IMAGE_IDS is still empty, show a message
    if not TEST_IMAGE_IDS:
        return render_template('index.html', view='test', current_image_id="N/A",
                               image_url="", mask_url="",
                               num_images=0, current_index=0,
                               message="No test images found or evaluated yet. Click 'Run Model' from the training view.")


    # Handle navigation (Next/Previous buttons) for test data
    action = request.args.get('action')
    if action == 'next':
        CURRENT_TEST_INDEX = (CURRENT_TEST_INDEX + 1) % len(TEST_IMAGE_IDS)
    elif action == 'prev':
        CURRENT_TEST_INDEX = (CURRENT_TEST_INDEX - 1 + len(TEST_IMAGE_IDS)) % len(TEST_IMAGE_IDS)
    
    # Store current test index in session
    session['current_test_index'] = CURRENT_TEST_INDEX

    current_image_id = TEST_IMAGE_IDS[CURRENT_TEST_INDEX]

    # URLs for the original test image and its predicted mask
    # We assume original test images are in PROCESSED_TEST_IMAGES_DIR
    image_url = url_for('serve_image', folder='test_images', filename=f"{current_image_id}.png")
    predicted_mask_url = url_for('serve_image', folder='flask_predictions', filename=f"{current_image_id}_pred_mask.png")

    session['show_test_results'] = True # Keep this flag active while in test results view

    return render_template('index.html',
                           view='test', # Indicate that we are in the test results view
                           current_image_id=current_image_id,
                           image_url=image_url,
                           mask_url=predicted_mask_url, # For test view, mask_url is predicted mask
                           num_images=len(TEST_IMAGE_IDS),
                           current_index=CURRENT_TEST_INDEX + 1) # +1 for 1-based indexing in UI


# --- Run the Flask Application ---
# --- Run the Flask Application ---
if __name__ == '__main__':
    # Initialize global variables when the script starts
    init_app_globals()
    
    # Run the Flask app
    # host='0.0.0.0' makes it accessible from other devices on your network
    # debug=True enables debug mode (auto-reloads on code changes, shows detailed errors)
    app.run(debug=True, host='0.0.0.0', port=5001) # <--- Changed port to 5001