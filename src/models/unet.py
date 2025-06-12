import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF # Added: Import for TF.resize
import yaml                                  # Added: Import for YAML configuration loading
import os                                    # Added: Import for path manipulation

class DoubleConv(nn.Module):
    """(convolution => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # Reverse the list for upsampling path

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Adjust size if needed (e.g., due to odd input dimensions or ConvTranspose2d behavior)
            if x.shape != skip_connection.shape:
                # TF.resize expects (H, W) tuple for 'size' argument
                # x.shape is (N, C, H, W), so skip_connection.shape[2:] gives (H, W)
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

if __name__ == "__main__":
    # --- Example usage: Load configuration and test the model ---

    # Construct the path to the config.yaml file
    # Assumes config.yaml is in the project root, two levels up from src/models/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_path = os.path.join(project_root, 'config.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}. Please ensure it's in your project root.")

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract model parameters
    model_params = config['model']['params']
    in_channels = model_params['in_channels']
    out_channels = model_params['out_channels']
    features = model_params['features']

    # Extract training parameters for dummy input size
    training_params = config['training']
    dummy_input_size = training_params['patch_size'] # e.g., [256, 256]

    # Create a dummy input tensor based on config
    # Batch size 1, configured channels, configured patch size
    dummy_input = torch.randn((1, in_channels, dummy_input_size[0], dummy_input_size[1]))

    print(f"Instantiating UNET model with:")
    print(f"  in_channels: {in_channels}")
    print(f"  out_channels: {out_channels}")
    print(f"  features: {features}")

    # Instantiate the model
    model = UNET(in_channels=in_channels, out_channels=out_channels, features=features)

    # Set model to evaluation mode for testing (disables dropout, batch norm updates)
    model.eval()

    # Perform a forward pass without gradient calculation
    with torch.no_grad():
        preds = model(dummy_input)

    print(f"\nDummy Input shape: {dummy_input.shape}")
    print(f"Model Output shape: {preds.shape}")

    print("\nUNET model definition test complete.")