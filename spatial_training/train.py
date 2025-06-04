import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from model import LSTMRegressor
from dataset import LoadDataset
import os
import logging
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import random_split

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#HyperParameters

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-8
MOMENTUM = 0.999
AMP = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 30
NUM_WORKER = 1
TRAIN_DIR = "data/spatial_data/training_data"
PATH = "spatial_training/checkpoints"


def train_fn(loader, val_loader, model, optimizer, criterion, grad_scaler):
    loop = tqdm(loader)
    gradient_clipping: float = 1.0
    
    model.train()
    for idx, (data, label) in enumerate(loop):
        optimizer.zero_grad()

        data = data.to(DEVICE)
        label = label.to(DEVICE)

        #forward
        if torch.cuda.amp.autocast_mode:
            predictions = model(data)
            loss = criterion(predictions.unsqueeze(-1), label.float())
            
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

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            predictions = model(x)
            loss = criterion(predictions.unsqueeze(-1), label.float())


    print(
        loss / max(len(val_loader), 1)
    )

    
    torch.save(model.state_dict(), os.path.join(PATH, f'checkpoint_manual_depth_{loss}.pt'))



def main():

    model = LSTMRegressor(input_dim=2, hidden_dim=64, spatial_size = 1).to(DEVICE)
    # criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(),
                              lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    
    
    train_ds = LoadDataset(
                    training_dir=TRAIN_DIR
    )

    train_size = int(0.8 * len(train_ds))
    val_size = len(train_ds) - train_size

    train_dataset, val_dataset = random_split(train_ds, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size= BATCH_SIZE,
        shuffle= True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size= BATCH_SIZE,
        shuffle= False
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
    