import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
import os
from tqdm import tqdm
import numpy as np
import sys
import json
import datetime # Import datetime for timestamping
from torchvision import transforms as T

# Import MLflow
import mlflow
import mlflow.pytorch

# Add the project root to the sys.path to allow importing modules from src/
# Correct calculation for files inside 'src' folder (e.g., src/train.py)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__))) # Go up one level from 'src'
sys.path.insert(0, project_root) # Insert at the beginning to prioritize project modules

# Import custom modules
from src.data_loader.nucleus_dataset import NucleusDataset
from src.data_loader.transforms import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from src.models.unet import UNET
from src.utils.losses import get_loss_function
from src.utils.metrics import dice_coefficient

# Helper function to get device
def get_device(device_str: str):
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "mps" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")

def train_model():
    # 1. Load configuration
    config_path = os.path.join(project_root, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model']['params']
    training_config = config['training']
    logging_config = config['logging']

    # 2. Set up device
    device = get_device(training_config['device'])
    print(f"Using device: {device}")
    use_pin_memory = (device.type == 'cuda')

    # 3. Create experiment-specific directories
    # These paths are needed for MLflow artifacts and local checkpoints/results
    exp_name = logging_config.get('exp_name', 'default_experiment')
    checkpoint_dir = os.path.join(project_root, logging_config['checkpoint_dir'], exp_name)
    results_dir = os.path.join(project_root, logging_config['results_dir'], exp_name)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    print(f"Results will be saved to: {results_dir}")

    # --- MLflow Tracking Setup ---
    # Set the MLflow tracking URI (where your MLflow server is running or local 'mlruns' folder)
    # By default, it's ./mlruns. You can change this to a remote server if needed.
    # mlflow.set_tracking_uri("http://localhost:5000") # Uncomment if you have a remote server

    # Set MLflow experiment name
    mlflow.set_experiment(config['project_name']) # Using project_name from config as MLflow experiment

    # Start an MLflow run
    with mlflow.start_run(run_name=exp_name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")):
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"MLflow Artifact URI: {mlflow.active_run().info.artifact_uri}")

        # Log parameters to MLflow
        mlflow.log_param("epochs", training_config['epochs'])
        mlflow.log_param("batch_size", training_config['batch_size'])
        mlflow.log_param("learning_rate", training_config['learning_rate'])
        mlflow.log_param("loss_function", training_config['loss_function'])
        mlflow.log_param("val_split_ratio", training_config['val_split_ratio'])
        mlflow.log_param("patch_size", training_config['patch_size'])
        mlflow.log_param("patch_stride", training_config['patch_stride'])
        mlflow.log_param("device", training_config['device'])
        mlflow.log_param("model_in_channels", model_config['in_channels'])
        mlflow.log_param("model_out_channels", model_config['out_channels'])
        mlflow.log_param("model_features", model_config['features'])
        mlflow.log_param("experiment_name", exp_name) # Log the experiment name itself

        # 4. Initialize Model
        model = UNET(
            in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'],
            features=model_config['features']
        ).to(device)
        print(f"Model initialized: {model}")

        # 5. Define Loss Function and Optimizer
        loss_fn = get_loss_function(training_config['loss_function'])
        optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])
        print(f"Loss Function: {loss_fn.__class__.__name__}")
        print(f"Optimizer: {optimizer.__class__.__name__} with learning rate {training_config['learning_rate']}")

        # 6. Set up Data Loaders with Train/Validation Split
        data_transforms = T.Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(degrees=(0, 30)),
            ToTensor()
        ])

        full_dataset = NucleusDataset(
            image_dir=os.path.join(project_root, training_config['image_dir']),
            mask_dir=os.path.join(project_root, training_config['mask_dir']),
            patch_size=tuple(training_config['patch_size']),
            patch_stride=tuple(training_config['patch_stride']),
            transform=data_transforms
        )

        val_size = int(len(full_dataset) * training_config['val_split_ratio'])
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        print(f"Total patches in dataset: {len(full_dataset)}")
        print(f"Patches allocated for training: {len(train_dataset)}")
        print(f"Patches allocated for validation: {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config['batch_size'],
            shuffle=True,
            num_workers=training_config['num_workers'],
            pin_memory=use_pin_memory
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            num_workers=training_config['num_workers'],
            pin_memory=use_pin_memory
        )

        print(f"Train DataLoader initialized with {len(train_dataset)} patches and batch size {training_config['batch_size']}. Number of batches: {len(train_loader)}")
        print(f"Val DataLoader initialized with {len(val_dataset)} patches and batch size {training_config['batch_size']}. Number of batches: {len(val_loader)}")

        # 7. Training Loop
        print(f"\nTraining started for {training_config['epochs']} epochs.")
        best_val_dice = 0.0
        history = []

        for epoch in range(training_config['epochs']):
            # --- Training Phase ---
            model.train()
            train_loss = 0.0
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_config['epochs']} (Train)")

            for batch_idx, data in enumerate(loop):
                images = data['image'].to(device)
                masks = data['mask'].to(device)

                predictions = model(images)
                loss = loss_fn(predictions, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_train_loss = train_loss / len(train_loader)
            # Log training loss to MLflow
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            print(f"Epoch {epoch+1} finished. Avg Train Loss: {avg_train_loss:.4f}")

            # --- Validation Phase ---
            model.eval()
            val_loss = 0.0
            val_dice_scores = []
            with torch.no_grad():
                val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{training_config['epochs']} (Val)")
                for batch_idx, data in enumerate(val_loop):
                    images = data['image'].to(device)
                    masks = data['mask'].to(device)

                    predictions = model(images)
                    loss = loss_fn(predictions, masks)
                    val_loss += loss.item()

                    batch_dice = dice_coefficient(predictions, masks)
                    val_dice_scores.append(batch_dice.item())

                    val_loop.set_postfix(val_loss=loss.item(), dice=batch_dice.item())

            avg_val_loss = val_loss / len(val_loader)
            avg_val_dice = np.mean(val_dice_scores)
            # Log validation loss and dice to MLflow
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_dice", avg_val_dice, step=epoch)
            print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, Validation Dice: {avg_val_dice:.4f}")

            # --- Store metrics for history JSON ---
            history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_dice': avg_val_dice
            })

            # --- Checkpointing Logic ---
            if avg_val_dice > best_val_dice:
                print(f"Validation Dice improved from {best_val_dice:.4f} to {avg_val_dice:.4f}. Saving model...")
                best_val_dice = avg_val_dice
                
                checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model saved to {checkpoint_path}")
                
                # Log the best model as an MLflow artifact
                # mlflow.pytorch.log_model(model, "best_model", registered_model_name=f"{config['project_name']}_UNET_Best")
                # For state_dict only:
                mlflow.log_artifact(checkpoint_path, "best_model_state_dict")
                
            else:
                print(f"Validation Dice did not improve ({avg_val_dice:.4f} vs {best_val_dice:.4f}).")

            # Optional: Save checkpoint at regular intervals
            if (epoch + 1) % logging_config['save_interval'] == 0:
                regular_checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), regular_checkpoint_path)
                print(f"Saved regular checkpoint to {regular_checkpoint_path}")
                mlflow.log_artifact(regular_checkpoint_path, f"checkpoints/epoch_{epoch+1}")


        print("Training complete.")

        # --- Save training history ---
        results_file_path = os.path.join(results_dir, "training_history.json")
        with open(results_file_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Training history saved to {results_file_path}")

        # Log training history JSON as an MLflow artifact
        mlflow.log_artifact(results_file_path, "training_history")

    print(f"MLflow run finished. View results by running 'mlflow ui' in your terminal from {project_root}.")


if __name__ == "__main__":
    train_model()