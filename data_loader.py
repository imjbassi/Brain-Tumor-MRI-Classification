# data_loader.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class BrainMRIDataset(Dataset):
    """
    Custom Dataset for loading brain MRI images from subfolders.
    Assumes a directory structure where each class has its own folder.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all images, organized in subdirectories per class.
            transform (callable, optional): Optional transform to apply to an image.
        """
        self.root_dir = root_dir
        self.transform = transform
        # List class directories sorted alphabetically
        classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        classes.sort()
        self.classes = classes
        # Map class names to numeric labels
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        # Gather all image file paths and labels
        self.image_paths = []
        self.labels = []
        for cls_name in self.classes:
            class_dir = os.path.join(root_dir, cls_name)
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, filename))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # ensure 3 channels
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloaders(data_dir, batch_size=32, val_split=0.2, shuffle=True, num_workers=4):
    """
    Create train and validation DataLoader from a directory of images.
    """
    # Define transforms: resize and normalize as in PyTorch tutorials
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet means:contentReference[oaicite:5]{index=5}
                             [0.229, 0.224, 0.225])  # ImageNet stds
    ])
    # Load full dataset
    dataset = BrainMRIDataset(root_dir=data_dir, transform=transform)
    # Split into train/validation sets
    total_size = len(dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # Create DataLoaders (shuffle training data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, dataset.classes
