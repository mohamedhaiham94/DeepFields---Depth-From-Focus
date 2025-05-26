import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

from tqdm import tqdm
from model import UNET

#from utils import *
from integral_utils import *

import logging
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('spatial_training/runs')

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#HyperParameters

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-8
MOMENTUM = 0.999
AMP = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 3
NUM_EPOCHS = 5
NUM_WORKER = 1

IMG_WIDTH = 1024
IMG_HEIGHT = 1024

TRAIN_STD_IMAGE_DIR = "data/spatial_data/train/integral"
TRAIN_DEPTH_MASK_DIR = "data/spatial_data/train/Depth"

VAL_STD_IMAGE_DIR = "data/spatial_data/validate/integral"
VAL_DEPTH_MASK_DIR = "data/spatial_data/validate/Depth"


PATH = "spatial_training/checkpoints"
def train_fn(loader, val_loader, model, optimizer, criterion, grad_scaler, epoch):
    loop = tqdm(loader)
    gradient_clipping: float = 1.0
    
    model.train()
    train_running_loss = 0

    for idx, (data, label) in enumerate(loop):
        optimizer.zero_grad()

        data = data.to(DEVICE).float()
        label = label.to(DEVICE).float()
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
            #loss += dice_loss(F.sigmoid(predictions.squeeze(1)), label.float(), multiclass=False)
            train_running_loss += loss.item()

        #backward
        optimizer.zero_grad(set_to_none=True)
        #grad_scaler.scale(loss).backward()
        #grad_scaler.unscale_(optimizer)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        #grad_scaler.step(optimizer)
        #grad_scaler.update()
        loss.backward()
        optimizer.step()

        # print(loss.item())
        loop.set_postfix(loss = loss.item())
    
    train_loss = train_running_loss / (idx + 1)
    
    writer.add_scalar("Loss/train", train_loss, epoch)

    model.eval()
    dice_score = 0
    val_running_loss = 0

    with torch.no_grad():
        for idx, (x, y) in enumerate(val_loader):
            x = x.to(DEVICE).float()
            y = y.to(DEVICE).float()

            # mask_pred = (F.sigmoid(model(x)) > 0.5).float()
            # compute the Dice score
            #dice_score += dice_coeff(mask_pred.squeeze(1), y, reduce_batch_first=False)
            
            predictions = model(x)
            # print(x.shape)
            # print(y.unsqueeze(1).shape)
            # print(predictions.shape)
            # sdf
            loss = criterion(predictions.squeeze(1), y.float())

            img_grid_x = torchvision.utils.make_grid(x)
            img_grid_pred = torchvision.utils.make_grid((torch.sigmoid(predictions) > 0.15) * 255)
            img_grid_mask = torchvision.utils.make_grid(y.unsqueeze(1))
            # print(type(img_grid_y))
            writer.add_image(tag='input', img_tensor=img_grid_x, global_step=epoch)
            writer.add_image(tag='mask', img_tensor=img_grid_mask, global_step=epoch)
            writer.add_image(tag='predection', img_tensor=img_grid_pred, global_step=epoch)
            val_running_loss += loss.item()

    val_loss = val_running_loss / (idx + 1)
    print("-"*30)
    print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
    print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
    print("-"*30)

    #print(
    #    dice_score / max(len(val_loader), 1), loss.item()
    #)

    #score = dice_score / max(len(val_loader), 1)
    
    torch.save(model.state_dict(), os.path.join(PATH, f'checkpoint_integral_depth_{epoch}.pt'))



def main():

    train_transform = A.Compose([
        #A.RandomCrop(IMG_HEIGHT, IMG_WIDTH),
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
        #A.CenterCrop(IMG_HEIGHT, IMG_WIDTH),
        # A.Normalize(
        #     mean= [0, 0, 0,],
        #     std= [1, 1, 1],
        #     max_pixel_value=255.0
        # ),
        ToTensorV2()
    ])


    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(),
                              lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    
    
    train_loader, val_loader = get_loader(
        TRAIN_STD_IMAGE_DIR,
        TRAIN_DEPTH_MASK_DIR,
        VAL_STD_IMAGE_DIR,
        VAL_DEPTH_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        validate_transform
    )

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, val_loader, model, optimizer, criterion, grad_scaler, epoch)




        # logging.basicConfig(
        #     filename='logs/model_logs.log',  # log file path
        #     filemode='a',               # overwrite existing logs ('a' for append)
        #     level=logging.INFO,         # log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        #     format='%(asctime)s - %(levelname)s - %(message)s'
        # )
        # logging.info(f'Validation Accuracy: {accuracy:.4f}')


if __name__ == "__main__":
    main()
    