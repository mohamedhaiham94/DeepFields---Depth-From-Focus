import numpy as np
import os
from PIL import Image


def load_images_with_stats(std_folder):
        
    rgb_files = sorted([f for f in os.listdir(std_folder) if f.endswith('.tiff')])

    
    rgb_images = []
    
    for rgb_file in rgb_files:

        rgb_img = Image.open(os.path.join(std_folder, rgb_file))
        rgb_array = np.array(rgb_img)
        
        rgb_images.append(rgb_array)

    return rgb_images

stack = np.array(load_images_with_stats('data\spatial_data\scenes\scene_2\STD'))

stack_reshaped = stack.transpose(1, 2, 0).reshape(-1, 100)  # shape: (1024*1024, 100)

from sklearn.cluster import KMeans

n_clusters = 2  # You can tune this
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
labels = kmeans.fit_predict(stack_reshaped)  # shape: (1024*1024,)

label_image = labels.reshape(1024, 1024)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.imshow(label_image, cmap='tab10')  # tab10 gives distinct colors
plt.title('Clustered Image from Stack')
plt.axis('off')
plt.show()
