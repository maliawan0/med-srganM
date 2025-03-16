from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as ttf
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

class GAN_Data(Dataset):
    def __init__(self, hr_folder, transforms=None, scale_factor=4):
        super().__init__()
        self.hr_paths = [os.path.join(hr_folder, f) for f in os.listdir(hr_folder)]
        self.transforms = transforms
        self.scale_factor = scale_factor
        self.downsample = ttf.Resize((512 // scale_factor, 512 // scale_factor))  # 128x128
        self.blur = ttf.GaussianBlur(3, sigma=(0.1, 2.0))

    def __getitem__(self, idx):
        # Load HR image (512x512)
        hr_img = Image.open(self.hr_paths[idx]).convert('RGB')
        hr_img = ttf.ToTensor()(hr_img)  # [0, 1]
        
        # Generate LR image (128x128)
        lr_img = self.downsample(hr_img)
        lr_img = self.blur(lr_img)

        if self.transforms:
            hr_img = self.transforms(hr_img)
            lr_img = self.transforms(lr_img)

        return lr_img.to(device), hr_img.to(device)

    def __len__(self):
        return len(self.hr_paths)