import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from model import UNET
import numpy as np
import logging
import cv2
from PIL import Image
from model import ScalarSequenceClassifier, LSTMRegressor, TransformerModel, SimpleAutoencoder, AttentionLSTM, MLP
import os

#HyperParameters

LEARNING_RATE = 1e4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 1
NUM_WORKER = 1

IMG_WIDTH = 128
IMG_HEIGHT = 128


TEST_STD_IMAGE_DIR = "data/Giovanni_data/angular_STD/max_results"
TEST_ENTROPY_IMAGE_DIR = "data/Giovanni_data/entropy/max_results"

PATH = "spatial_training/checkpoints"

        
def load_images_stack(img_std_dir, img_entropy_dir, img_depth_mask_dir):

    img_std_files = sorted([f for f in os.listdir(img_std_dir) if f.endswith('.tiff')],
                        key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    img_entropy_files = sorted([f for f in os.listdir(img_entropy_dir) if f.endswith('.tiff')],
                        key=lambda x: int(x.split('_')[-1].split('.')[0]))
    

    img_depth_mask_files = sorted([f for f in os.listdir(img_depth_mask_dir) if f.endswith('.png')],
                        key=lambda x: int(x.split('_')[2]))
    
    # print(img_depth_mask_files[69].split('_'), img_entropy_files[69], img_std_files[69])
    std_images = []
    entropy_images = []
    depth_mask_images = []
    
    for std_file, entropy_file, depth_file in zip(img_std_files, img_entropy_files, img_depth_mask_files):

        std_img = Image.open(os.path.join(img_std_dir, std_file))
        std_array = np.array(std_img)
        
        entropy_img = Image.open(os.path.join(img_entropy_dir, entropy_file))
        entropy_array = np.array(entropy_img)
        
        depth_img = Image.open(os.path.join(img_depth_mask_dir, depth_file)).convert('L')
        depth_array = np.array(depth_img)
        
        std_images.append(std_array)
        entropy_images.append(entropy_array)
        depth_mask_images.append(depth_array)

    return std_images, entropy_images, depth_mask_images
    
def test_fn(model, spatial_size=1):

    std_images, entropy_images, depth_images = load_images_stack(r'data/spatial_data/testing/max_results/STD',
                                                                 r'data/spatial_data/testing/max_results/Entropy',
                                                                  r'data\spatial_data\testing\Depth' )
    std_stack = np.stack(std_images)  # shape: (num_images, height, width, channels)
    entropy_stack = np.stack(entropy_images)  # shape: (num_images, height, width)
    depth_stack = np.stack(depth_images) / 255
    # slice_2x2x100 = std_stack[69:70, 0:2, 0:2] # z, y, x
    results = np.zeros((1, 1024, 1024))

    # # Iterate over spatial dimensions
    # #std_stack.shape[1] - spatial_size
    # for x in tqdm(range(std_stack.shape[1] - spatial_size), desc="Processing X"):
    #     for y in tqdm(range(std_stack.shape[2] - spatial_size), desc="Processing Y", leave=False):

    #         std_vector = std_stack[5, y:y+spatial_size, x:x+spatial_size] 
    #         entropy_vector = entropy_stack[5, y:y+spatial_size, x:x+spatial_size] 
    #         depth_vector = depth_stack[5, y:y+spatial_size, x:x+spatial_size] 

    #         # input_vector = np.stack((std_vector, entropy_vector), axis=-1)
    #         avg_std = std_stack[5].mean()
    #         input_vector = np.append(std_vector, avg_std) 

    #         input_vector = torch.from_numpy(input_vector).unsqueeze(0)
    #         depth_vector = torch.from_numpy(depth_vector).unsqueeze(0)


    #         input_vector = input_vector.to(DEVICE)
    #         depth_vector = depth_vector.to(DEVICE)

    #         # label = label.unsqueeze(1).to(DEVICE)
    #         # input_vector = input_vector.unsqueeze(2)

    #         #forward
    #         if torch.cuda.amp.autocast_mode:
    #             input_vector = input_vector.view(input_vector.size(0), -1)

    #             predictions = model(input_vector[:, 0].unsqueeze(-1))

    #             # predictions = model(input_vector) #.permute(0, 3, 1, 2)).squeeze(0)
    #             predictions = torch.sigmoid(predictions) > 0.5
    #             # print(predictions, depth_vector)
                
    #             predictions = predictions.squeeze(-1).cpu().detach().numpy()

    #             results[:, y:y+spatial_size, x:x+spatial_size] = predictions 

    idx = 0
    H, W = std_stack[43].shape
    flat_std = std_stack[43].flatten()
    flat_avg = [std_stack[43].mean()] * len(flat_std)

    # Each pixel gets a feature vector: [std, std]
    feats = np.stack([flat_std, flat_avg], axis=1)  # shape: (H*W, 2)

    # print(torch.from_numpy(feats).shape)
    # sdf
    preds = model(torch.from_numpy(feats).cuda())

    # preds = model(torch.from_numpy(feats).cuda())

    preds = torch.sigmoid(preds) > 0.8

    preds = preds.squeeze(-1).cpu().detach().numpy()

    # Reshape predictions back to image shape
    results = preds.reshape(H, W).astype(int)

    print(results.shape)
    # for i in results:
    cv2.imwrite(f'spatial_training/out/{44}.png', results.astype(int) * 255)
    idx +=1 
                    



def main():

    # model = LSTMRegressor(input_dim=1, hidden_dim=16, spatial_size = 1).to(DEVICE)
    # model = TransformerModel().to(DEVICE)
    model = MLP(2).to(DEVICE)

    model.load_state_dict(torch.load(os.path.join(PATH, 'checkpoint_33.pt'), weights_only=True))
    model.to('cuda')
    model.eval()


    for _ in range(NUM_EPOCHS):
        test_fn(model)

        # check_accuracy(val_loader, model, DEVICE)
        
        # logging.basicConfig(
        #     filename='logs/model_logs.log',  # log file path
        #     filemode='a',               # overwrite existing logs ('a' for append)
        #     level=logging.INFO,         # log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        #     format='%(asctime)s - %(levelname)s - %(message)s'
        # )
        # logging.info(f'Validation Accuracy: {accuracy:.4f}')


if __name__ == "__main__":
    main()
    