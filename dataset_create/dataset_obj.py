import os
from PIL import Image
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
import numpy as np
class CustomImageDataset(Dataset):
    def __init__(self, input_path, shape_path, labels_path, labels_shape_path, transform=None):
        self.imgs = np.reshape(np.load(input_path), np.load(shape_path))
        self.labels = np.reshape(np.load(labels_path), np.load(labels_shape_path))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #image = Image.fromarray(self.imgs[idx])
        image = self.imgs[idx]
        label = self.labels[idx]
        image = torch.tensor(image).float()
        image = torch.unsqueeze(image, 0)
        label = torch.tensor(label).float()
        label = torch.unsqueeze(label, 0)
        if self.transform:
            image = self.transform(image)
        return image, label
