import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import tifffile
import cv2

class LoadDataset(Dataset):
    
    def __init__(self, img_std_dir, img_depth_mask_dir, transform) -> None:
        super().__init__()

        self.img_integral_dir = img_std_dir
        self.img_depth_mask_dir = img_depth_mask_dir
        self.transform = transform

        self.images = os.listdir(img_std_dir)
    

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):

        image_integral_path = os.path.join(self.img_integral_dir, self.images[index])
        
        if self.img_depth_mask_dir is not None:
            img_number = self.images[index].split('_')[-1].split('.')[0]
            # For Depth folder
            depth_image_name = f'depth_camera_{img_number}_0001.png'
            
            # For manual_depth folder
            #depth_image_name = f'{img_number}.png'
            
            depth_path = os.path.join(self.img_depth_mask_dir, depth_image_name)
            depth = np.array(Image.open(depth_path))

            # For manual_depth folder
            mask = depth / 255
            #mask = 1 - mask

        int_image = np.array(Image.open(image_integral_path))
        
        image = int_image[:, :, :3]
 
        #image = np.stack((std_image, entropy_image), axis=0)
        # image = (image - image.min()) / (image.max() - image.min())
        # print(image.max(), image.min())
        # dfg
        #image = image.transpose(1, 2, 0)
        # plt.imshow(image)
        # plt.show()
        # tifffile.imwrite('output_2channels.tiff', image)


        
        # cv2.imwrite(f'spatial_training/out/{4}.png', mask * 255)

        if self.img_depth_mask_dir is None:
            augmentation = self.transform(image = image)

            image = augmentation["image"]

            return image/255


        if self.transform is not None:
            augmentation = self.transform(image = image, mask = mask)

            image = augmentation["image"]
            mask = augmentation["mask"]

        return image/255, mask

if __name__ == "__main__":

    dataloader = LoadDataset("data/spatial_data/train/integral", 
                             "data/spatial_data/train/manual_depth", 
                             None)
    print(dataloader.__len__())
    image, mask = dataloader.__getitem__(5)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the first image in the first subplot
    axes[0].imshow(image)
    axes[0].axis('off')  # Turn off axes

    # Display the second image in the second subplot
    axes[1].imshow(mask)
    axes[1].axis('off')  # Turn off axes

    # Show the plot
    plt.show()