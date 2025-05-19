import torch
import torch.optim as optim
import torch.nn as nn

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
from model import UNET

import numpy as np

from utils import *

import logging

import cv2

#HyperParameters

LEARNING_RATE = 1e4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 1
NUM_WORKER = 1

IMG_WIDTH = 128
IMG_HEIGHT = 128


TEST_STD_IMAGE_DIR = "data/spatial_data/test/STD"
TEST_ENTROPY_IMAGE_DIR = "data/spatial_data/test/Entropy"
TEST_DEPTH_MASK_DIR = "data/spatial_data/test/manual_depth"

PATH = "spatial_training/checkpoints"

def test_fn(loader, model):
    loop = tqdm(loader)

    for idx, (data, label) in enumerate(loop):
        data = data.to(DEVICE)
        # data = (data - data.min()) / (data.max() - data.min())
        
        label = label.unsqueeze(1).to(DEVICE)

        #forward
        if torch.cuda.amp.autocast_mode:
            predictions = model(data).squeeze(0)
            predictions = torch.sigmoid(predictions) > 0.5

            predictions = predictions.permute(1, 2, 0).cpu().detach().numpy()
            cv2.imwrite(f'spatial_training/out/{idx}.tiff', predictions.astype(np.float32))
            

        # print(loss.item())


def main():

    test_transform = A.Compose([
        # A.RandomCrop(IMG_HEIGHT, IMG_WIDTH),
        # A.Normalize(
        #     mean= [0, 0, 0,],
        #     std= [1, 1, 1],
        #     max_pixel_value=255.0
        # ),
        ToTensorV2()
    ])


    model = UNET(in_channels=2, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(PATH, 'checkpoint.pt'), weights_only=True))
    model.to('cuda')
    model.eval()

    test_loader = get_test_loader(
        TEST_STD_IMAGE_DIR,
        TEST_ENTROPY_IMAGE_DIR,
        TEST_DEPTH_MASK_DIR,
        test_transform
    )

    for _ in range(NUM_EPOCHS):
        test_fn(test_loader, model)

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
    