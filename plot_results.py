import matplotlib.pyplot as plt
import json
import os
import sys
import yaml
import numpy as np # Import numpy for mean calculation

# Add the project root to the sys.path
# Correct calculation for files inside src/ (e.g., src/plot_results.py)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__))) # Go up one level from 'src'
sys.path.insert(0, project_root) # Insert at the beginning to prioritize project modules

# Modified to accept specific results directory
def plot_training_results(config: dict, results_dir: str):
    print(f"\n--- Plotting Training Results for Experiment: {os.path.basename(results_dir)} ---")

    history_file_path = os.path.join(results_dir, 'training_history.json') # Look in the specific results_dir

    if not os.path.exists(history_file_path):
        print(f"Error: Training history file not found at {history_file_path}. Skipping plotting.")
        print("Please ensure the training step has completed successfully and generated this file.")
        return

    try:
        with open(history_file_path, 'r') as f:
            history_list = json.load(f) # history is now a list of dictionaries
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {history_file_path}: {e}")
        return
    
    if not history_list:
        print(f"Warning: Training history file {history_file_path} is empty or malformed. No data to plot.")
        return

    # Extract data from the list of dictionaries
    epochs = [h['epoch'] for h in history_list]
    train_losses = [h['train_loss'] for h in history_list]
    val_losses = [h['val_loss'] for h in history_list]
    val_dice_scores = [h['val_dice'] for h in history_list] # Get validation Dice scores

    # Plotting Loss
    plt.figure(figsize=(14, 6)) # Adjust figure size for two subplots
    
    plt.subplot(1, 2, 1) # First subplot for Loss
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plotting Validation Dice Coefficient
    plt.subplot(1, 2, 2) # Second subplot for Dice
    plt.plot(epochs, val_dice_scores, label='Validation Dice Coefficient', color='green')
    plt.title('Validation Dice Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.grid(True)
    
    # Add a note about the best validation Dice coefficient
    if val_dice_scores:
        best_val_dice = np.max(val_dice_scores)
        plt.text(0.05, 0.95, f'Best Val Dice: {best_val_dice:.4f}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='cyan', alpha=0.5))


    # Display the plots
    plt.tight_layout()
    plt.show()
    print("Plots displayed successfully.")

if __name__ == "__main__":
    # This block is only for direct testing of plot_results.py
    # In a real scenario, this main() would be called by run.py
    _config_path = os.path.join(project_root, 'config.yaml')
    if os.path.exists(_config_path):
        with open(_config_path, 'r') as f:
            _config = yaml.safe_load(f)
        
        # Use 'exp_name' from the new config structure
        _experiment_name = _config['logging']['exp_name']
        # Use 'results_dir' (base) from the new config structure
        _results_base_dir = os.path.join(project_root, _config['logging']['results_dir'])
        _results_dir = os.path.join(_results_base_dir, _experiment_name)
        
        print("Running plot_results.py directly. Please ensure training_history.json exists in:")
        print(_results_dir)
        plot_training_results(_config, _results_dir)
    else:
        print(f"config.yaml not found at {_config_path} for direct plot_results.py run. Exiting.")