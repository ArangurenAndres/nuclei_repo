import os
from train import train_model
from evaluate import evaluate_model
from src.utils.image_processing import preprocess_and_save_train_data, preprocess_and_save_test_images

def run_pipeline():
    # 1. Preprocess training and test data
    print("Starting the preprocessing and training pipeline...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))

    # Paths for training data
    RAW_TRAIN_PATH = os.path.join(project_root, 'data', 'raw', 'stage1_train')
    PROCESSED_TRAIN_IMAGES_PATH = os.path.join(project_root, 'data', 'processed', 'train_images')
    PROCESSED_TRAIN_MASKS_PATH = os.path.join(project_root, 'data', 'processed', 'train_masks')
    if not ( (os.path.exists(PROCESSED_TRAIN_IMAGES_PATH) and len(os.listdir(PROCESSED_TRAIN_IMAGES_PATH)) > 0) and
             (os.path.exists(PROCESSED_TRAIN_MASKS_PATH) and len(os.listdir(PROCESSED_TRAIN_MASKS_PATH)) > 0) ):

        print("\n--- Processing Training Data ---")
        preprocess_and_save_train_data(RAW_TRAIN_PATH, PROCESSED_TRAIN_MASKS_PATH, PROCESSED_TRAIN_IMAGES_PATH)
        print("Training data preprocessing finished.")
    else:
        print("Processed training data already found and complete. Skipping preprocessing.")


    # Paths for test data (NEW)
    RAW_TEST_PATH = os.path.join(project_root, 'data', 'raw', 'stage1_test')
    PROCESSED_TEST_IMAGES_PATH = os.path.join(project_root, 'data', 'processed', 'test_images') # New folder for processed test images
    if not (os.path.exists(PROCESSED_TEST_IMAGES_PATH) and len(os.listdir(PROCESSED_TEST_IMAGES_PATH)) > 0):
        print("\n--- Processing Test Data ---")
        preprocess_and_save_test_images(RAW_TEST_PATH, PROCESSED_TEST_IMAGES_PATH)
        print("Test data preprocessing finished.")
    else:
        print("Processed test data already found and complete. Skipping preprocessing.")
    print("\nAll preprocessing tasks complete!")
    
    # 2. Train the model
    train_model()

    # 3. Evaluate model
    evaluate_model()
    
if __name__ == "__main__":
    run_pipeline()