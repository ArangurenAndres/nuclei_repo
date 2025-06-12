import torch
import matplotlib.pyplot as plt
import os
import json
import seaborn as sns
import pandas as pd


def dice_coefficient(predictions, targets, smooth=1e-6):
    """
    Calculates the Dice Coefficient (F1-score) between predicted and ground truth masks.

    """
    # Apply sigmoid to predictions to convert logits to probabilities
    # Then, convert probabilities to binary predictions (0 or 1) by thresholding
    # A common threshold is 0.5, but for Dice calculation directly on probabilities,
    # it's usually better to keep them as probabilities before calculating intersection/union.
    # However, for a "coefficient" we often imply a binary comparison.
    # Let's keep it consistent with DiceLoss internal sigmoid, but also add a threshold concept.

    # For Dice Coeff, we usually want a binary prediction. Let's threshold at 0.5 after sigmoid.
    preds = torch.sigmoid(predictions)
    preds = (preds > 0.5).float() # Convert to binary (0 or 1) based on threshold

    # Flatten tensors
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum()

    # Dice = (2 * intersection) / (union)
    dice = (2. * intersection + smooth) / (union + smooth)

    return dice


def plot_training_history(history_json_path: str, save_path: str = None):
    """
    Plots the training and validation loss, and validation Dice Coefficient over epochs
    in separate subplots.

    Args:
        history_json_path (str): Path to the training history JSON file.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    if not os.path.exists(history_json_path):
        print(f"Error: History file not found at {history_json_path}")
        return

    with open(history_json_path, 'r') as f:
        history = json.load(f)

    if not history:
        print("Error: Training history is empty. No data to plot.")
        return

    df = pd.DataFrame(history)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # 1 row, 2 columns for plots

    # Plot 1: Train Loss and Validation Loss
    sns.lineplot(x='epoch', y='train_loss', data=df, label='Train Loss', marker='o', ax=axes[0])
    sns.lineplot(x='epoch', y='val_loss', data=df, label='Validation Loss', marker='x', ax=axes[0])
    axes[0].set_title('Training and Validation Loss Over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (Dice Loss)')
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Validation Dice Coefficient
    sns.lineplot(x='epoch', y='val_dice', data=df, label='Validation Dice Coefficient', color='green', linestyle='--', marker='s', ax=axes[1])
    axes[1].set_title('Validation Dice Coefficient Over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Coefficient')
    axes[1].set_ylim(0, 1) # Dice Coefficient ranges from 0 to 1
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
    plt.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

