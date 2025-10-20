# src/data/cifar_text.py
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

class CIFARTextDataset(Dataset):
    def __init__(self, root='./data', train=True, download=True, transform=None):
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
        self.ds = datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        img, label = self.ds[idx]
        text = CIFAR10_CLASSES[label]  # simple text
        return img, text, label
