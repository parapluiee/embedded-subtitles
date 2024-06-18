import os
from PIL import Image
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
class CustomImageDataset(Dataset):
    def __init__(self, labels, img_dir, transform=None):
        self.img_labels = labels
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.index[idx])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx]
        if self.transform:
            ten_image = self.transform(image)
        label = torch.tensor(label.values.astype(float))
        return ten_image, label
