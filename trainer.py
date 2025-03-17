from generator import Generator
from discriminator import Discriminator
from feature_extractor import FeatureExtractor
from dataset import GAN_Data
from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as ttf
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# Update with your HR dataset folder
hr_folder = r"C:\Users\Ali\Desktop\dataset\ct"


# Initialize models
gen = Generator().to(device)
disc = Discriminator().to(device)
feature_extractor = FeatureExtractor().to(device)
feature_extractor.eval()

# Data augmentation
transforms = ttf.Compose([
    ttf.RandomHorizontalFlip(0.5),
    ttf.RandomVerticalFlip(0.5),
    ttf.RandomRotation((-15, 15)),
])

# Dataset and dataloader
dataset = GAN_Data(hr_folder, transforms=transforms)
train_dl = DataLoader(dataset, batch_size=1, shuffle=True)

# Optimizers and loss
optimizer_G = optim.Adam(gen.parameters(), lr=1e-3, weight_decay=1e-5)
optimizer_D = optim.Adam(disc.parameters(), lr=1e-4, weight_decay=1e-5)
loss_function = torch.nn.L1Loss().to(device)
gan_loss = torch.nn.BCEWithLogitsLoss().to(device)
scaler = torch.amp.GradScaler(device_type='cuda', enabled=True)



def fit(gen, disc, feature_extractor, train_dl, epochs, optimizer_G, optimizer_D, scaler, loss_function, gan_loss):
    t_loss_G, t_loss_D = [], []
    for epoch in tqdm(range(epochs)):
        e_loss_G, e_loss_D = [], []
        for data in train_dl:
            lr_img, hr_img = data
            valid = torch.ones((1, 2), dtype=torch.float32, device=device)
            fake = torch.zeros((1, 2), dtype=torch.float32, device=device)

            # Train Generator
            with torch.cuda.amp.autocast():
                pred_hr = gen(lr_img)
                content_loss = loss_function(pred_hr, hr_img)
                pred_features = feature_extractor(pred_hr)
                hr_features = feature_extractor(hr_img)
                feature_loss = sum(loss_function(pf, hf) for pf, hf in zip(pred_features, hr_features))
                pred_real = disc(hr_img.detach(), lr_img)
                pred_fake = disc(pred_hr, lr_img)
                gan_loss_num = gan_loss(pred_fake - pred_real.mean(0, keepdim=True), valid)
                loss_G = content_loss * 0.1 + feature_loss * 0.1 + gan_loss_num

            optimizer_G.zero_grad()
            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            scaler.update()
            e_loss_G.append(loss_G.item())

            # Train Discriminator
            with torch.cuda.amp.autocast():
                pred_real = disc(hr_img, lr_img)
                pred_fake = disc(pred_hr.detach(), lr_img)
                loss_real = gan_loss(pred_real - pred_fake.mean(0, keepdim=True), valid)
                loss_fake = gan_loss(pred_fake - pred_real.mean(0, keepdim=True), fake)
                loss_D = (loss_real + loss_fake) / 2

            optimizer_D.zero_grad()
            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)
            scaler.update()
            e_loss_D.append(loss_D.item())

        t_loss_G.append(sum(e_loss_G) / len(e_loss_G))
        t_loss_D.append(sum(e_loss_D) / len(e_loss_D))
        print(f"Epoch {epoch+1}/{epochs} | G Loss: {t_loss_G[-1]:.4f} | D Loss: {t_loss_D[-1]:.4f}")

        # Save models
        torch.save(gen.state_dict(), f"./gen_epoch_{epoch+1}.pth")
        torch.save(disc.state_dict(), f"./disc_epoch_{epoch+1}.pth")

    return t_loss_G, t_loss_D


print("pls woek")


# Start training
fit(gen, disc, feature_extractor, train_dl, 100, optimizer_G, optimizer_D, scaler, loss_function, gan_loss)