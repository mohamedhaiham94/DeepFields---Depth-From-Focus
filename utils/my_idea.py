
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, classification_report, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


size = (1024,1024)  # For example, resizing images to 200x200

script_dir = os.path.dirname(os.path.realpath(__file__))

#os.chdir(script_dir)

print(f"Current working directory: {os.getcwd()}")

def load_images_with_stats(rgb_folder, alpha_folder, depth_file_name, size):
        
    rgb_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith('.png')],
                        key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    alpha_files = sorted([f for f in os.listdir(alpha_folder) if f.endswith('.png')],
                        key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    depth_img = Image.open(os.path.join(depth_file_name))
    
    rgb_images = []
    alpha_images = []
    
    for rgb_file, alpha_file in zip(rgb_files, alpha_files):

        rgb_img = Image.open(os.path.join(rgb_folder, rgb_file))
        rgb_img = rgb_img.resize(size)  # Resize to the specified size
        rgb_array = np.array(rgb_img)
        
        alpha_img = Image.open(os.path.join(alpha_folder, alpha_file)).convert('L')
        alpha_img = alpha_img.resize(size)  # Resize to the specified size

        alpha_array = np.array(alpha_img)
        
        rgb_images.append(rgb_array)
        alpha_images.append(alpha_array)

    return rgb_images, alpha_images, depth_img


'''
def angular_STD(rgb_images,alpha_images):
        height, width, _ = rgb_images[0].shape
            
        for y in range(height):
            for x in range(width):

                pixel_values = np.array([np.mean(img[y, x]) for img, alpha in zip(rgb_images, alpha_images) if alpha[y, x] == 1]) /1
                plt.figure(figsize=(14, 8))
                plt.plot(np.array(pixel_values).reshape(-1, 1))
                plt.title(f'Distribution of Image # {y, x}')
                plt.grid(True)
                plt.savefig(f'out/{y}_{x}.png')
                plt.close()
                
                if pixel_values.shape[0] == 0:
                    continue
                
               
        return pixel_values
'''

import numpy as np
# Defining the softmax function
def softmax(values):

    # Computing element wise exponential value
    exp_values = np.exp(values)

    # Computing sum of these values
    exp_values_sum = np.sum(exp_values)

    # Returing the softmax output.
    return exp_values/exp_values_sum

def angular_STD(rgb_images, alpha_images, depth_img):
    # Stack images into numpy arrays
    rgb_stack = np.stack(rgb_images) / 255  # shape: (num_images, height, width, channels)
    alpha_stack = np.stack(alpha_images)  # shape: (num_images, height, width)
    depth_img = np.array(depth_img) / 255
    
    # Mask RGB images using alpha
    valid_mask = alpha_stack == 1  # valid pixels only
    valid_rgb = np.where(valid_mask[..., None], rgb_stack, np.nan)

    pixel_vectors = valid_rgb.reshape(100, -1, 3)  # Flatten spatial dimensions (100 x 1,048,576 x 3)
    depth_vector = depth_img.reshape(-1, 1)

    ### To Know x, y from an index use 
    '''
    width = 1024
    index = 100

    y = index // width  # row index
    x = index % width   # column index

    print(f"Coordinates: (x={x}, y={y})")
    '''
    
    # To Know index from x, y use
    '''
    y = 496
    x = 374
    width = 1024

    index = y * width + x
    print(index)
    '''

    
    split_idx = int(0.8 * (1024 * 1024))

    train_data = pixel_vectors[:, :split_idx, :]  # (83,886,080, 3)
    train_y = depth_vector[:split_idx]
    val_data = pixel_vectors[:, split_idx:, :]    # (20,971,520, 3)
    val_y = depth_vector[split_idx:]
    return train_data, val_data, train_y, val_y
    # print(pixel_vectors[:, 630166, :])
    # print(depth_vector[630166])
    # g
    # vector = pixel_vectors[:, 1000000, 0]
    # # non_nan_values = vector[~np.isnan(vector).any(axis=1)]
    # plt.imshow(np.array(vector).reshape(10, 10))
    # plt.show()
    # print(non_nan_values.shape)
    # sdf
    # Compute mean across color channels first
    # rgb_means = np.nanmean(valid_rgb, axis=-1)  # shape: (num_images, height, width)

    # Red channel shape : (100, 1024, 1024, 1) valid_rgb[:, :, :, 0]
    # Green channel shape : (100, 1024, 1024, 1) valid_rgb[:, :, :, 1]
    # Blue channel shape : (100, 1024, 1024, 1) valid_rgb[:, :, :, 2]


    # c = rgb_means[:, 590, 417]
    # #print(c[np.array(np.logical_not(np.isnan(c)))])
    # plt.figure(figsize=(14, 8))
    


    # weights = 1 / (rgb_stack[:, 493, 273] + 1e-7)
    # weights /= np.sum(weights)  # Normalize weights
    
    # # Predicted soft depth
    # depth_layers = np.arange(100)
    # predicted_depth = np.sum(depth_layers * weights)
    # print(predicted_depth)
    # average = []
    # for i in range(100):
    #     average.append(np.mean(rgb_stack[i,:,:]))
    # features = {
    #     'min_std_value': np.min(rgb_stack[:, 493, 273] * weights),
    #     'min_std_index': np.argmin(rgb_stack[:, 493, 273] * weights),
    #     'mean_std': np.mean(rgb_stack[:, 493, 273] * weights),
    #     'max_std': np.max(rgb_stack[:, 493, 273] * weights),
    #     'range_std': np.ptp(rgb_stack[:, 493, 273] * weights),  # peak-to-peak range
    # }
        
    # print(features)


    
    # plt.plot(np.array(c[np.array(np.logical_not(np.isnan(c)))]).reshape(-1, 1))#
    # plt.imshow(np.array(rgb_means[:, 590, 417]).reshape(10, 10))
    
    # weights = 1 / (rgb_stack[:, 493, 274] + 1e-7)
    # weights /= np.sum(weights)  # Normalize weights
    # plt.plot(np.array(rgb_stack[:, 493, 274] * weights).reshape(-1, 1))
    
    
    #plt.plot(np.array(rgb_stack[:, 493, 273]).reshape(-1, 1))

    #plt.plot(np.array(average * weights).reshape(-1, 1))

    # i = 0
    # plt.title(f'Distribution of Image # {i}')
    # plt.grid(True)
    # plt.show()
    # Compute angular STD (standard deviation along the angular dimension - axis 0)
    # angular_std_R = np.nanstd(valid_rgb[:, :, :, 0], axis=0)  # shape: (height, width)
    # angular_std_G = np.nanstd(valid_rgb[:, :, :, 1], axis=0)  # shape: (height, width)
    # angular_std_B = np.nanstd(valid_rgb[:, :, :, 2], axis=0)  # shape: (height, width)
    # stacked = np.dstack((angular_std_R, angular_std_G, angular_std_B))
    # max_std = np.max(stacked, axis=2)
    # cv2.imwrite('out/test.tiff', max_std.astype(np.float32))
    # # print(angular_std[533, 89])
    # sdf
    # return angular_std



def create_or_clear_folder(folder_path):
    if os.path.exists(folder_path):
       
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
       
        os.makedirs(folder_path)


class MLPRegressor(nn.Module):
    def __init__(self):
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Output is a single depth value
        )

    def forward(self, x):
        return self.model(x)

model = MLPRegressor()



'''
min_folder = r'./min_results'  # Folder to save min images
mean_folder = r'./mean_results' # Folder to save mean images
max_folder = r'./max_results'  # Folder to save max images

create_or_clear_folder(min_folder)
create_or_clear_folder(mean_folder)
create_or_clear_folder(max_folder)
'''
#alpha_folders = [d for d in os.listdir('variance') if d.startswith('alpha')]


# for i in range(len(alpha_folders)):
    
#     rbg_path = r"./variance/shifted_images_{}".format(i)
#     alpha_path = r"./variance/alpha_{}".format(i)

#     rgb_images, alpha_images = load_images_with_stats(rbg_path, alpha_path, size=size)


for i in range(1):

    # rbg_path = "data/raw/TopDown/max/STD".format(4)
    # alpha_path = "data/raw/TopDown/max/STD".format(4)

   
    rbg_path = r"data/raw/TopDown/variance/shifted_images_{}".format(3)
    alpha_path = r"data/raw/TopDown/variance/alpha_{}".format(3)
    depth_path = r"data/raw/Depth/depth_camera_{}_0001.png".format(3 + 1)
    
    rgb_images, alpha_images, depth_img = load_images_with_stats(rbg_path, alpha_path, depth_path, size=size)


    
    X_train, X_test, y_train, y_test = angular_STD(rgb_images,alpha_images, depth_img)

    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    
    dataset = TensorDataset(X_train.float(), y_train.unsqueeze(0).expand(100, -1, -1).float())
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    # Training loop
    for epoch in range(100):  # number of epochs
        for batch_X, batch_y in loader:
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 1 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    gg   