import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import tifffile
import cv2

class LoadDataset(Dataset):
    
    def __init__(self, img_std_dir, img_entropy_dir, img_depth_mask_dir, transform) -> None:
        super().__init__()

        self.img_std_dir = img_std_dir
        self.img_entropy_dir = img_entropy_dir
        self.img_depth_mask_dir = img_depth_mask_dir
        self.transform = transform

        self.images = os.listdir(img_std_dir)
    

    def __len__(self):
        return len(self.images)
    def load_images_stack(self):
    
        img_std_files = sorted([f for f in os.listdir(self.img_std_dir) if f.endswith('.tiff')],
                            key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        img_entropy_files = sorted([f for f in os.listdir(self.img_entropy_dir) if f.endswith('.tiff')],
                            key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        img_depth_mask_files = sorted([f for f in os.listdir(self.img_depth_mask_dir) if f.endswith('.png')],
                            key=lambda x: int(x.split('_')[2]))
        
        print(img_depth_mask_files[69].split('_'), img_entropy_files[69], img_std_files[69])
        std_images = []
        entropy_images = []
        depth_mask_images = []
        
        for std_file, entropy_file, depth_file in zip(img_std_files, img_entropy_files, img_depth_mask_files):

            std_img = Image.open(os.path.join(self.img_std_dir, std_file))
            std_array = np.array(std_img)
            
            entropy_img = Image.open(os.path.join(self.img_entropy_dir, entropy_file))
            entropy_array = np.array(entropy_img)
            
            depth_img = Image.open(os.path.join(self.img_depth_mask_dir, depth_file)).convert('L')
            depth_array = np.array(depth_img)
            
            std_images.append(std_array)
            entropy_images.append(entropy_array)
            depth_mask_images.append(depth_array)

        return std_images, entropy_images, depth_mask_images
    
    def __getitem__(self, index):
        print(index)
        std_images, entropy_images, depth_images = self.load_images_stack()
        
        std_stack = np.stack(std_images)  # shape: (num_images, height, width, channels)
        entropy_stack = np.stack(entropy_images)  # shape: (num_images, height, width)
        depth_stack = np.stack(depth_images) / 255
        slice_2x2x100 = std_stack[:, 0:1, 0:1] # z, y, x

        print(std_stack.shape, entropy_stack.shape, depth_stack.shape, slice_2x2x100.shape)
        print(slice_2x2x100)
        sdf
        image_std_path = os.path.join(self.img_std_dir, self.images[index])
        image_entropy_path = os.path.join(self.img_entropy_dir, self.images[index].replace("variance", "entropy"))
        
        if self.img_depth_mask_dir is not None:
            img_number = self.images[index].split('_')[-1].split('.')[0]
            # For Depth folder
            depth_image_name = f'depth_camera_{img_number}_0001.tiff'
            
           
            depth_path = os.path.join(self.img_depth_mask_dir, depth_image_name)

            distance = 0.03
            depth = np.array(Image.open(depth_path))
            
            # For Depth folder
            mask = np.where(depth <= distance, 1, 0)
            
            # For manual_depth folder
            mask = depth / 255
            mask = 1 - mask


        std_image = np.array(Image.open(image_std_path))
        entropy_image = np.array(Image.open(image_entropy_path))
        
        image = np.stack((std_image, entropy_image), axis=0)
        # image = (image - image.min()) / (image.max() - image.min())
        # print(image.max(), image.min())
        # dfg
        image = image.transpose(1, 2, 0)
        # plt.imshow(image)
        # plt.show()
        # tifffile.imwrite('output_2channels.tiff', image)


        


        if self.img_depth_mask_dir is None:
            return image


        if self.transform is not None:
            augmentation = self.transform(image = image, mask = mask)

            image = augmentation["image"]
            mask = augmentation["mask"]

        return image, mask

if __name__ == "__main__":

    dataloader = LoadDataset("data/spatial_data/scene_1/STD", 
                             "data/spatial_data/scene_1/Entropy", 
                             "data/spatial_data/scene_1/Depth", 
                             None)

    image, mask = dataloader.__getitem__(2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the first image in the first subplot
    axes[0].imshow(image)
    axes[0].axis('off')  # Turn off axes

    # Display the second image in the second subplot
    axes[1].imshow(mask)
    axes[1].axis('off')  # Turn off axes

    # Show the plot
    plt.show()