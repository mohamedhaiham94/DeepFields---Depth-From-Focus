
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import tifffile
import cv2
import pickle 
from tqdm import tqdm

class ComputeTrainingPickels:
    def __init__(self, img_std_dir, 
                 img_entropy_dir, 
                 img_depth_mask_dir, 
                 out_dir_positive, 
                 out_dir_negative, 
                 spatial_size,
                 scene_number) -> None:
        super().__init__()

        self.img_std_dir = img_std_dir
        self.img_entropy_dir = img_entropy_dir
        self.img_depth_mask_dir = img_depth_mask_dir
        self.out_dir_positive = out_dir_positive
        self.out_dir_negative = out_dir_negative
        self.spatial_size = spatial_size
        self.scene_number = scene_number

        self.saving_pickles(self.spatial_size, self.scene_number)
        
    def load_images_stack(self):
    
        img_std_files = sorted([f for f in os.listdir(self.img_std_dir) if f.endswith('.tiff')],
                            key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        img_entropy_files = sorted([f for f in os.listdir(self.img_entropy_dir) if f.endswith('.tiff')],
                            key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        img_depth_mask_files = sorted([f for f in os.listdir(self.img_depth_mask_dir) if f.endswith('.png')],
                            key=lambda x: int(x.split('_')[2]))
        
        # print(img_depth_mask_files[69].split('_'), img_entropy_files[69], img_std_files[69])
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
    
    def save_pickle(self, data, path):
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def saving_pickles(self, spatial_size, scene_number):
        '''
            spatial_size: (int) is the size of the slicing stack 1x1 or 2x2 or nxn
        '''
        std_images, entropy_images, depth_images = self.load_images_stack()
        std_stack = np.stack(std_images)  # shape: (num_images, height, width, channels)
        entropy_stack = np.stack(entropy_images)  # shape: (num_images, height, width)
        depth_stack = np.stack(depth_images) / 255
        # slice_2x2x100 = std_stack[69:70, 0:2, 0:2] # z, y, x
        
        avg_std = std_stack[scene_number].mean()

        # Iterate over spatial dimensions
        for x in tqdm(range(std_stack.shape[1] - spatial_size), desc="Processing X"):
            for y in tqdm(range(std_stack.shape[2] - spatial_size), desc="Processing Y", leave=False):
                std_vector = std_stack[scene_number, y:y+spatial_size, x:x+spatial_size] 
                entropy_vector = entropy_stack[scene_number, y:y+spatial_size, x:x+spatial_size] 
                depth_vector = depth_stack[scene_number, y:y+spatial_size, x:x+spatial_size] 


                # input_vector = np.stack((std_vector, entropy_vector), axis=-1)
                input_vector = np.append(std_vector, avg_std)

                # print(input_vector.shape, input_vector[69, :, :, :], depth_vector.shape)
                # sdfsdf
                # Save as pickle


                if sum(depth_vector) > 0:
                    file_path = os.path.join(self.out_dir_positive, f"input_vector_{x}_{y}_scene_{scene_number}.pkl")
                    with open(file_path, 'wb') as f:
                        pickle.dump(input_vector, f)
                        
                    # file_path = os.path.join(self.out_dir_positive, f"entropy_vector_{x}_{y}_scene_{scene_number}.pkl")
                    # with open(file_path, 'wb') as f:
                    #     pickle.dump(entropy_vector, f)
                        
                    file_path = os.path.join(self.out_dir_positive, f"depth_vector_{x}_{y}_scene_{scene_number}.pkl")
                    with open(file_path, 'wb') as f:
                        pickle.dump(depth_vector, f)
                else:
                    file_path = os.path.join(self.out_dir_negative, f"input_vector_{x}_{y}_scene_{scene_number}.pkl")
                    with open(file_path, 'wb') as f:
                        pickle.dump(input_vector, f)
                        
                    # file_path = os.path.join(self.out_dir_negative, f"entropy_vector_{x}_{y}_scene_{scene_number}.pkl")
                    # with open(file_path, 'wb') as f:
                    #     pickle.dump(entropy_vector, f)
                        
                    file_path = os.path.join(self.out_dir_negative, f"depth_vector_{x}_{y}_scene_{scene_number}.pkl")
                    with open(file_path, 'wb') as f:
                        pickle.dump(depth_vector, f)
                    
if __name__ == "__main__":


    for i in range(47, 100):
        # Create unique training data folder for this iteration
        training_data_dir = f"data/spatial_data/train_data/training_data_{i+1}"
        positive_dir = os.path.join(training_data_dir, "positive")
        negative_dir = os.path.join(training_data_dir, "negative")
        
        # Create directories if they don't exist
        os.makedirs(positive_dir, exist_ok=True)
        os.makedirs(negative_dir, exist_ok=True)
        
        # Run the data loader with current iteration's paths
        dataloader = ComputeTrainingPickels(
            "data/spatial_data/scenes/scene_1/STD",
            "data/spatial_data/scenes/scene_1/Entropy",
            "data/spatial_data/scenes/scene_1/depth",
            positive_dir,
            negative_dir,
            spatial_size=1,
            scene_number=i
        )