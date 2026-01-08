import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # 1. Feature Extraction Block
        # Conv Layer 1: Sees simple patterns (edges, lines)
        # Input: 3 channels (RGB), Output: 32 feature maps
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces size by half
        
        # Conv Layer 2: Sees more complex patterns (shapes, textures)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Conv Layer 3: Sees high-level features (parts of objects)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # 2. Classification Block
        # We need to flatten the 3D maps into a 1D vector
        # Image is resized to 224x224. 
        # After 3 pools (divide by 2 three times): 224 -> 112 -> 56 -> 28
        # Final shape: 128 channels * 28 * 28
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(0.5) # Prevents overfitting

    def forward(self, x):
        # Pass through Conv 1
        x = self.pool(F.relu(self.conv1(x)))
        
        # Pass through Conv 2
        x = self.pool(F.relu(self.conv2(x)))
        
        # Pass through Conv 3
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten: keep batch size (x.size(0)), flatten the rest
        x = x.view(x.size(0), -1) 
        
        # Pass through Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Output layer (raw scores/logits)
        
        return x

def get_model(num_classes=2):
    return SimpleCNN(num_classes=num_classes)
