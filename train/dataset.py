import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class XRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
                               Structure should be:
                               root_dir/
                                   NORMAL/
                                   PNEUMONIA/
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.images = []
        self.labels = []

        # Load images
        for idx, label_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, label_name)
            if not os.path.exists(class_dir):
                continue
                
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, file_name))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
