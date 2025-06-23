from minisom import MiniSom
import numpy as np
from torch.utils.data import DataLoader
import torch
from dataloader import PatchDataset
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from scipy.stats import wasserstein_distance


# # 1. Use the PatchDataset to Extract Patches
# patch_size = 64
# stride = 32

# # Define your image stack (list of image filenames)
# image_stack = [img for img in os.listdir('dataset') if img.endswith(('.png', '.jpg', '.jpeg', '.tif'))]

# # Create the dataset and DataLoader
# dataset = PatchDataset(image_stack=image_stack, patch_size=patch_size, stride=stride)
# dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

# # 2. Flatten Patches for SOM Training
# # Gather all patches into a single array for SOM training
# all_patches = []

# for batch in dataloader:
#     patches, _ = batch  # We only need the patches, not labels
#     patches = patches.numpy().reshape(patches.size(0), -1)  # Flatten each patch
#     all_patches.append(patches)

# all_patches = np.vstack(all_patches)  # Combine into one matrix (N x D)

# # 3. Train the SOM
# som_dim = 10  # SOM grid size (10x10)
# input_len = all_patches.shape[1]  # Feature vector length (flattened patch size)
# som = MiniSom(x=som_dim, y=som_dim, input_len=input_len, sigma=1.0, learning_rate=0.5)

# print("Training SOM...")
# som.train_random(data=all_patches, num_iteration=1000)  # Train SOM on patches
# print("SOM Training Complete.")

# # 4. Use the SOM for Classification
# # Classify new patches by computing BMU distance
# results = []

import cv2
patches = cv2.imread(r'stack20143.tif', -1)
patches = cv2.resize(patches, (64, 64), interpolation = cv2.INTER_LINEAR)
# patches = (patches - np.min(patches)) / (np.max(patches) - np.min(patches))
patches = patches.flatten()


patches2 = cv2.imread(r'stack20143_focus.tif', -1)
patches2 = cv2.resize(patches2, (64, 64), interpolation = cv2.INTER_LINEAR)
# patches2 = (patches2 - np.min(patches2)) / (np.max(patches2) - np.min(patches2))
patches2 = patches2.flatten()


print(wasserstein_distance(patches2, patches))

# patch = patches.flatten()
# bmu = som.winner(patch)  # Get BMU coordinates
# distance = np.linalg.norm(patch - som.get_weights()[bmu])  # Compute distance to BMU
# results.append((bmu, distance))

# print(results)
# 5. Threshold for Focus/Out-of-Focus Classification
# threshold = 0.5  # Set an appropriate threshold based on validation
# classified_results = ["Focused" if dist > threshold else "Out-of-Focus" for _, dist in results]
# print(classified_results)
print("Classification Complete.")
