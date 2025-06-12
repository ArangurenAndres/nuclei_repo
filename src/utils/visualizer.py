import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Tuple, Union

def visualize_batch(images: List[np.ndarray], masks: List[np.ndarray],
                    predictions: Union[List[np.ndarray], None] = None,
                    num_display: int = 4,
                    save_path: Union[str, None] = None,
                    title_prefix: str = ""):
    """
    Visualizes a batch of images, their ground truth masks, and optionally predictions.
    """
    num_items = min(len(images), num_display)
    if num_items == 0:
        print("No items to display.")
        return

    has_predictions = predictions is not None
    cols = 2 if not has_predictions else 3

    # --- FIX IS HERE: Reduced the figsize factor from 4.5 to 3.0 ---
    # You can adjust this factor (e.g., to 2.5 or 3.5) based on your screen size
    fig, axes = plt.subplots(num_items, cols, figsize=(cols * 3.0, num_items * 3.0))
    # --- END FIX ---

    if num_items == 1:
        axes = np.array([axes])

    for i in range(num_items):
        img = images[i]
        mask = masks[i]
        sample_label = f"Sample {i+1}"

        # Original Image
        ax_img = axes[i, 0]
        ax_img.imshow(img, cmap='gray')
        ax_img.set_title(f"{sample_label}\nOriginal Image")
        ax_img.axis('off')

        # Ground Truth Mask
        ax_mask = axes[i, 1]
        ax_mask.imshow(mask, cmap='gray', vmin=0, vmax=255)
        ax_mask.set_title(f"{sample_label}\nGround Truth Mask")
        ax_mask.axis('off')

        if has_predictions:
            pred = predictions[i]
            ax_pred = axes[i, 2]
            ax_pred.imshow(pred, cmap='gray', vmin=0, vmax=255)
            ax_pred.set_title(f"{sample_label}\nPredicted Mask")
            ax_pred.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.suptitle(title_prefix, y=1.0, fontsize=16)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()
    plt.close(fig)