import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.util import view_as_windows
from PIL import Image
import os

class PatchDataset(Dataset):
    def __init__(self, image, patch_size, stride = 4, depth = 6):
        
        image_array = image

        # Extract 4x4 patches
        # patches, patch_grid_shape = self.extract_patches(image_array, patch_size=patch_size, stride = stride)
        patches, patch_grid_shape = self.extract_patches_with_axial_context(image_array, patch_size=patch_size, stride = stride, depth_context = depth)

        print(f"patch shape is = {patches.shape}")
        
        self.patches = torch.tensor(patches, dtype=torch.float32)

    def extract_patches(self, image_array, patch_size, stride):
        """
        Extract non-overlapping patches of size patch_size x patch_size from a 2D image.
        Returns a 4D numpy array: (num_patches, 1, patch_size, patch_size)
        """
        # Ensure input is a numpy array
        if isinstance(image_array, torch.Tensor):
            image_array = image_array.numpy()

        # Extract patches
        patches = view_as_windows(image_array, (patch_size, patch_size), step=stride)


        num_rows, num_cols, _, _ = patches.shape
        patches = patches.reshape(-1, patch_size, patch_size)

        # Add channel dimension
        patches = patches[:, np.newaxis, :, :]  # Shape: (N, 1, 4, 4)

        return patches, (num_rows, num_cols)

    def extract_patches_with_axial_context(self, std_folder, patch_size, stride, depth_context=6):
        """
        Extract patches with axial (depth) context from a 3D volume.
        
        Parameters:
            std_folder: path
            patch_size: spatial size (int)
            stride: spatial stride (int)
            depth_context: number of adjacent axial slices (must be odd)

        Returns:
            patches: (N, depth_context, patch_size, patch_size)
            grid_shape: (num_rows, num_cols, num_depth_slices)
        """
        std_stack = self.load_images_with_stats(std_folder)
        print(std_stack.shape)
        stack_slice = std_stack[:depth_context, :, :]  # Shape: (1024, 1024, 6)
        stack_slice = np.transpose(stack_slice, (1, 2, 0))  # (H, W, D)

        # Use view_as_windows to extract patches in (H, W) with full depth_context
        patches = view_as_windows(stack_slice, (patch_size, patch_size, depth_context), step=(stride, stride, depth_context))
        num_rows, num_cols, _, _, _, _ = patches.shape
        # print(num_rows, num_cols)


        # Reshape to (N, patch_size, patch_size, depth_context)
        patches = patches.reshape(-1, patch_size, patch_size, depth_context)

        return patches, (num_rows, num_cols)

    def load_images_with_stats(self, std_folder):
            
        std_files = sorted([f for f in os.listdir(std_folder) if f.endswith('.tiff')],
                            key=lambda x: int(x.split('_')[-1].split('.')[0]))

        
        std_images = []
        
        for std_file in std_files:

            std_img = Image.open(os.path.join(std_folder, std_file))
            std_array = np.array(std_img)
            
            std_images.append(std_array)

        return np.stack(std_images)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]
    
    def get_patches(self):
        return self.patches

