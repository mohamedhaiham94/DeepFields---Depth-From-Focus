import pandas as pd 
import numpy as np
import os 
import cv2 
from tqdm import tqdm

from utils import *

class ProcessingDatasetClass:
    
    def __init__(self, dataset_path, depth_path):
        self.dataset_path = dataset_path
        self.depth_path = depth_path
        
    def pixel_depth_pair_to_dataframe(self, width, height):
        """
        Generates and saves (pixel value, depth value) pairs for a given dataset.

        This function processes all images in the specified dataset (Entropy, STD, or Spatial STD),
        generates corresponding pixel-depth value pairs, and saves the results for each image
        separately as a CSV file.

        Note:
            The (x, y) coordinates have been flipped to match the format used by ImageJ, 
            allowing for easier result verification.

        Args:
            width (int): Image width
            height (int): Image height
        """
        depth_data = sorted(os.listdir(self.depth_path),key=numericalSort)
        
        image_data_len = len(depth_data)
        
        
        record_modes  = ['TopDown'] #Circular
        metrics  = ['max'] #, 'min', 'mean'
        feature_types  = ['Entropy'] #Spatial_STD, Entropy
        
        for record_mode in tqdm(record_modes, desc="Record Modes"):
            for metric in tqdm(metrics, desc=f"{record_mode} - Metrics", leave=False):
                for feature_type in tqdm(feature_types, desc=f"{record_mode}-{metric} Features", leave=False):
                    
                    image_data = sorted(os.listdir(os.path.join(self.dataset_path, record_mode, metric, feature_type)),key=numericalSort)
                    for i in tqdm(range(image_data_len), desc=f"{record_mode}-{metric}-{feature_type} Images", leave=False):
                        image = cv2.imread(os.path.join(self.dataset_path, record_mode, metric, feature_type, image_data[i]), cv2.IMREAD_UNCHANGED)
                        depth = cv2.imread(os.path.join(self.depth_path, depth_data[i]), 0) / 255
                        data = []
                        avg_value = np.mean(image)
                        
                        for y in range(width):
                            for x in range(height):    
                                 data.append(
                                    {"x": y, "y": x, metric+'_'+feature_type: image[x,y], "depth_value":depth[x, y], "avg_value":avg_value}
                                )
                        df = pd.DataFrame(data)
                        file_name = f"Image_{i + 1}_{metric}_{feature_type}.csv"
                        df.to_csv(os.path.join(r'data\processed', record_mode, metric, feature_type, file_name), index=False)

        


if __name__ == "__main__":
    obj = ProcessingDatasetClass(r'data\raw', r'data\raw\Depth')
    obj.pixel_depth_pair_to_dataframe(1024, 1024)
    print('Hi')