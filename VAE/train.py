import torch
import torch.nn as nn
from model import VAE
from utils import *
from dataloader import PatchDataset
from torch.utils.data import DataLoader, ConcatDataset
import os
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np

# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 128
condition_dim = 1

for i in range(19, 101):
    vae = VAE(latent_dim = latent_dim, depth = i).to(device)
    # vae = VAE(img_size = 8, nb_channels = 1, z_dim=32, latent_img_size = latent_dim).to(device)

    print(vae)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    DIR = 'dataset'

    # Assuming patches is a NumPy array of shape (N, 1, 4, 4)
    image_path = f"data/Testing-scenes/Scene_2_dense_1_spacing_row_0.30_col_damage_2_seed_225/max_results/max_variance_image_{i}.tiff"

    # image = Image.open(image_path)

    # cv2.imwrite('test.tiff', (np.array(image) * np.array(depth) / 255).astype(np.float32))
    # sdf
    # patch_dataset = PatchDataset((np.array(image) * (1 - (np.array(depth) / 255))).astype(np.float32), 4, 4)
    patch_dataset = PatchDataset('data/Testing-scenes/Scene_2_dense_1_spacing_row_0.30_col_damage_2_seed_225/max_results/STD', 4, 4, i)

    # depth_dataset = PatchDataset(np.array(depth), 4, 4)

    # # # Assuming patches is a NumPy array of shape (N, 1, 4, 4)
    # image_path2 = "data/raw/TopDown/max/max_variance_image_77.tif"
    # patch_dataset2 = PatchDataset(image_path, 4, 4)

    # combined_dataset = ConcatDataset([patch_dataset2, patch_dataset])

    patch_dataset = np.transpose(patch_dataset, (0, 3, 1, 2))  # (N, 6, 4, 4)
    patch_dataloader = DataLoader(patch_dataset, batch_size=64, shuffle=True)
    # depth_dataloader = DataLoader(depth_dataset, batch_size=64, shuffle=False)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


    # Training loop with tqdm
    epochs = 10
    for epoch in range(epochs):
        vae.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        # Wrap the dataloader with tqdm for a progress bar
        loop = tqdm(patch_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for images in loop:  # Full-resolution images as input
            images = images.to(device)
            
            optimizer.zero_grad()
            loss, recon_mb, _, _ = vae.step(
                images
            )

            
            # recon, mu, logvar = vae(images)

            # loss, loss_dict = vae_loss_function(recon, images, mu, logvar)
            (loss).backward()
            optimizer.step()

            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)

            total_loss += loss.item()
            # total_recon += loss_dict['recon_loss']
            # total_kl += loss_dict['kl_div']

            # Update tqdm description with the current loss

            loop.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(patch_dataloader)}")
        scheduler.step()

    # Save the final model
    torch.save(vae.state_dict(), f"VAE/checkpoint/vae_scene_2_128_layer_{i}.pth")
    print(f"Model saved successfully! for Layer {i}")