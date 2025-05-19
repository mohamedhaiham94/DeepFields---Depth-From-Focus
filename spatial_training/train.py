import torch
import torch.optim as optim
import torch.nn as nn

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

from tqdm import tqdm
from model import UNET

from utils import *

import logging
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#HyperParameters

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-8
MOMENTUM = 0.999
AMP = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 5
NUM_WORKER = 1

IMG_WIDTH = 256
IMG_HEIGHT = 256

TRAIN_STD_IMAGE_DIR = "data/spatial_data/train/STD"
TRAIN_ENTROPY_IMAGE_DIR = "data/spatial_data/train/Entropy"
TRAIN_DEPTH_MASK_DIR = "data/spatial_data/train/manual_depth"

VAL_STD_IMAGE_DIR = "data/spatial_data/validate/STD"
VAL_ENTROPY_IMAGE_DIR = "data/spatial_data/validate/Entropy"
VAL_DEPTH_MASK_DIR = "data/spatial_data/validate/manual_depth"


PATH = "spatial_training/checkpoints"
def train_fn(loader, val_loader, model, optimizer, criterion, grad_scaler):
    loop = tqdm(loader)
    gradient_clipping: float = 1.0
    
    model.train()
    for idx, (data, label) in enumerate(loop):
        optimizer.zero_grad()

        data = data.to(DEVICE)
        label = label.to(DEVICE)
        # print(data[0,0,::].shape, label[0,::].shape)
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # # Display the first image in the first subplot
        # axes[0].imshow(data[0,0,::].cpu().detach().numpy())
        # axes[0].axis('off')  # Turn off axes

        # # Display the second image in the second subplot
        # axes[1].imshow(label[0,::].cpu().detach().numpy())
        # axes[1].axis('off')  # Turn off axes

        # # Show the plot
        # plt.show()
        
        # sdf
        #forward
        if torch.cuda.amp.autocast_mode:
            predictions = model(data)
            loss = criterion(predictions.squeeze(1), label.float())
            loss += dice_loss(F.sigmoid(predictions.squeeze(1)), label.float(), multiclass=False)
            
        #backward
        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        
        # print(loss.item())
        loop.set_postfix(loss = loss.item())
    
    
    model.eval()
    dice_score = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            mask_pred = (F.sigmoid(model(x)) > 0.5).float()
            # compute the Dice score
            dice_score += dice_coeff(mask_pred.squeeze(1), y, reduce_batch_first=False)

    print(
        dice_score / max(len(val_loader), 1)
    )

    
    
    torch.save(model.state_dict(), os.path.join(PATH, 'checkpoint.pt'))



def main():

    train_transform = A.Compose([
        A.RandomCrop(IMG_HEIGHT, IMG_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=1.0),
        # A.Normalize(
        #     mean= [0, 0, 0,],
        #     std= [1, 1, 1],
        #     max_pixel_value=255.0
        # ),
        ToTensorV2()
    ])

    validate_transform = A.Compose([
        A.CenterCrop(IMG_HEIGHT, IMG_WIDTH),
        # A.Normalize(
        #     mean= [0, 0, 0,],
        #     std= [1, 1, 1],
        #     max_pixel_value=255.0
        # ),
        ToTensorV2()
    ])


    model = UNET(in_channels=2, out_channels=1).to(DEVICE)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(),
                              lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    
    
    train_loader, val_loader = get_loader(
        TRAIN_STD_IMAGE_DIR,
        TRAIN_ENTROPY_IMAGE_DIR,
        TRAIN_DEPTH_MASK_DIR,
        VAL_STD_IMAGE_DIR,
        VAL_ENTROPY_IMAGE_DIR,
        VAL_DEPTH_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        validate_transform
    )

    for _ in range(NUM_EPOCHS):
        train_fn(train_loader, val_loader, model, optimizer, criterion, grad_scaler)




        # logging.basicConfig(
        #     filename='logs/model_logs.log',  # log file path
        #     filemode='a',               # overwrite existing logs ('a' for append)
        #     level=logging.INFO,         # log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        #     format='%(asctime)s - %(levelname)s - %(message)s'
        # )
        # logging.info(f'Validation Accuracy: {accuracy:.4f}')


if __name__ == "__main__":
    main()
    