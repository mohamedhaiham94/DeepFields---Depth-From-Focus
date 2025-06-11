import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from model import LSTMRegressor, TransformerModel, SimpleAutoencoder, ScalarSequenceClassifier, AttentionLSTM, MLP
from dataset import LoadDataset
import os
import logging
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.cuda.amp import autocast, GradScaler

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#HyperParameters
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_EPOCHS = 15
NUM_WORKER = 1
TRAIN_DIR = "data/spatial_data/train_data/training_data_44/positive"
PATH = "spatial_training/checkpoints"


def train_fn(loader, val_loader, model, optimizer, criterion):
    loop = tqdm(loader)

    
    model.train()
    for idx, (data, label) in enumerate(loop):

        data = data.to(DEVICE)
        label = label.to(DEVICE)

        #forward
        # print(data[:, 0].shape, data.shape)
        # sdf
        predictions = model(data)

        # loss = criterion(predictions.unsqueeze(-1), label.float())
        # print(label.squeeze(-1).squeeze(-1), predictions)
        
        loss = criterion(predictions, label.squeeze(-1).float())

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # grad_scaler.scale(loss).backward()
        # grad_scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        # grad_scaler.step(optimizer)
        # grad_scaler.update()
        
        # print(loss.item())
        # if idx % 10 == 0:
        loop.set_postfix(loss=loss.item())    
    
    model.eval()

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            # x = x.view(x.size(0), -1)

            predictions = model(x)
            loss = criterion(predictions, y.squeeze(-1).float())


    print(
        loss / max(len(val_loader), 1)
    )
    
    
    
    torch.save(model.state_dict(), os.path.join(PATH, f'checkpoint_44.pt'))

def main():
    print(DEVICE)
    # model = LSTMRegressor(input_dim=1, hidden_dim=16, spatial_size = 1).to(DEVICE)
    # model = TransformerModel().to(DEVICE)
    model = MLP(2).to(DEVICE)
    
    
    
    # criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    pos_weight = 0.0028 #28.021620671713926
    pos_weight = torch.tensor(pos_weight).cuda()

    criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(),
                              lr=LEARNING_RATE)
    
    
    train_ds = LoadDataset(
                    training_dir=TRAIN_DIR
    )

    train_size = int(0.80 * len(train_ds))
    val_size = len(train_ds) - train_size

    generator1 = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_ds, [train_size, val_size], generator=generator1)

    train_loader = DataLoader(
        train_dataset,
        batch_size= BATCH_SIZE,
        shuffle= True,
        num_workers=NUM_WORKER,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size= BATCH_SIZE,
        shuffle= False
    )
    
    
    
    for _ in range(NUM_EPOCHS):
        train_fn(train_loader, val_loader, model, optimizer, criterion)




        # logging.basicConfig(
        #     filename='logs/model_logs.log',  # log file path
        #     filemode='a',               # overwrite existing logs ('a' for append)
        #     level=logging.INFO,         # log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        #     format='%(asctime)s - %(levelname)s - %(message)s'
        # )
        # logging.info(f'Validation Accuracy: {accuracy:.4f}')


if __name__ == "__main__":
    main()
    