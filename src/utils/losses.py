import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    It's often used with BCEWithLogitsLoss.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply sigmoid to predictions to get probabilities, as Dice Loss operates on probabilities/binary masks
        inputs = torch.sigmoid(inputs)

        # Flatten label and prediction tensors for calculation
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    """
    Combination of Binary Cross-Entropy Loss and Dice Loss.
    Commonly used for medical image segmentation.
    """
    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss() # Uses raw logits (no sigmoid needed before)
        self.dice_loss = DiceLoss(smooth=smooth) # DiceLoss expects raw logits and applies sigmoid internally
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.bce_weight * bce + self.dice_weight * dice

def get_loss_function(loss_name: str):
    """
    Factory function to retrieve a loss function by name.
    """
    if loss_name == "BCEWithLogitsLoss":
        # PyTorch's built-in BCEWithLogitsLoss
        return nn.BCEWithLogitsLoss()
    elif loss_name == "DiceLoss":
        return DiceLoss()
    elif loss_name == "CombinedLoss":
        return CombinedLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")

if __name__ == "__main__":
    # Example usage and test for losses.py
    print("Testing custom loss functions...")

    # Dummy predictions (logits) and target
    preds_logits = torch.randn(4, 1, 256, 256) # Batch, Channels, H, W
    targets = torch.randint(0, 2, (4, 1, 256, 256)).float() # Binary target (0 or 1)

    # Test BCEWithLogitsLoss (built-in)
    bce_loss_fn = get_loss_function("BCEWithLogitsLoss")
    bce_loss = bce_loss_fn(preds_logits, targets)
    print(f"BCEWithLogitsLoss: {bce_loss.item():.4f}")

    # Test DiceLoss
    dice_loss_fn = get_loss_function("DiceLoss")
    dice_loss = dice_loss_fn(preds_logits, targets)
    print(f"DiceLoss: {dice_loss.item():.4f}")

    # Test CombinedLoss
    combined_loss_fn = get_loss_function("CombinedLoss")
    combined_loss = combined_loss_fn(preds_logits, targets)
    print(f"CombinedLoss (BCE+Dice): {combined_loss.item():.4f}")

    print("Loss functions tested successfully.")