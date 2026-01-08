import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import sys

# Add project root to path so we can import our modules
sys.path.append(os.getcwd())

from app.model import get_model
from train.dataset import XRayDataset

def train_model(data_dir, num_epochs=5, batch_size=32):
    print("--- Starting Training Session ---")
    
    # 1. Image Preprocessing (Transforms)
    # Data Augmentation for training helps the model generalize better
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Validation transforms should be simpler (no random crops/flips)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load Dataset
    # Initially load the whole dataset
    full_dataset = XRayDataset(data_dir)
    if len(full_dataset) == 0:
        print("Error: No data found in", data_dir)
        return

    # 3. Split into Training (80%) and Validation (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # Apply specific transforms to each subset
    class ApplyTransform(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
        
        def __len__(self):
            return len(self.subset)

    train_dataset = ApplyTransform(train_subset, train_transform)
    val_dataset = ApplyTransform(val_subset, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Initialize Model, Loss Function, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=2).to(device)
    
    # CrossEntropyLoss is the standard for classification (Normal vs Pneumonia)
    criterion = nn.CrossEntropyLoss()
    
    # Adam is a more modern "engine" that often learns faster and more reliably
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. The Training Loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # Tell model it's learning
            else:
                model.eval()  # Tell model it's just being tested

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in (train_loader if phase == 'train' else val_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                # Reset the gradients
                optimizer.zero_grad()

                # Forward Pass (Predict)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward Pass (Learn)
                    if phase == 'train':
                        loss.backward() # Calculate errors
                        optimizer.step() # Fix weights

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / (train_size if phase == 'train' else val_size)
            epoch_acc = running_corrects.double() / (train_size if phase == 'train' else val_size)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # 6. Save the trained "brain"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/best_model.pth")
    print("\nTraining complete! Model saved to models/best_model.pth")

if __name__ == "__main__":
    train_model("data", num_epochs=15)
