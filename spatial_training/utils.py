import os, shutil
import torch
from dataset import LoadDataset
from torch.utils.data import DataLoader
import random
import torch
from torch import Tensor
import torch.nn.functional as F


STD_PATH = 'data/spatial_data/train/STD'
ENTROPY_PATH = 'data/spatial_data/train/Entropy'
DEPTH_PATH = 'data/spatial_data/train/Depth'

std_imgs = os.listdir(STD_PATH)

depth_mask = os.listdir(DEPTH_PATH)

std_imgs.sort()

validate_ratio = int(len(std_imgs) * 0.2)
files_to_move = random.sample(std_imgs, validate_ratio)

# for file_name in files_to_move:
#     # moving the std data from training to validation
#     shutil.move(os.path.join(STD_PATH, file_name), os.path.join(STD_PATH.replace('train', 'validate'), file_name))
    
#     # moving the entropy data from training to validation
#     entropy_name = file_name.replace("variance", "entropy")
#     shutil.move(os.path.join(ENTROPY_PATH, entropy_name), os.path.join(ENTROPY_PATH.replace('train', 'validate'), entropy_name))
    
#     # moving the depth data from training to validation
#     img_number = file_name.split('_')[-1].split('.')[0]
#     depth_image_name = f'depth_camera_{img_number}_0001.tiff'
#     shutil.move(os.path.join(DEPTH_PATH, depth_image_name), os.path.join(DEPTH_PATH.replace('train', 'validate'), depth_image_name))



def get_loader(
        img_std_dir, 
        img_entropy_dir, 
        img_depth_mask_dir, 
        val_std_dir, 
        val_entropy_dir, 
        val_depth_mask_dir, 
        batch_size, 
        train_transform, 
        validate_transform):
    
    train_ds = LoadDataset(
        img_std_dir= img_std_dir,
        img_entropy_dir= img_entropy_dir,
        img_depth_mask_dir= img_depth_mask_dir,
        transform= train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size= batch_size,
        shuffle= True
    )

    val_ds = LoadDataset(
        img_std_dir= val_std_dir,
        img_entropy_dir= val_entropy_dir,
        img_depth_mask_dir= val_depth_mask_dir,
        transform= validate_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size= batch_size,
        shuffle= False
    )


    return train_loader, val_loader


def get_test_loader(
        img_std_dir, 
        img_entropy_dir, 
        img_depth_mask_dir, 
        test_transform
        ):
    
    test_ds = LoadDataset(
        img_std_dir= img_std_dir,
        img_entropy_dir= img_entropy_dir,
        img_depth_mask_dir= img_depth_mask_dir,
        transform= test_transform
    )

    test_loader = DataLoader(
        test_ds,
        batch_size= 1,
        shuffle= True
    )

    return test_loader



def check_accuracy(loader, model, device = "cuda"):
    num_correct = 0
    num_pixels = 0

    model.eval()
    dice_score = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            mask_pred = (F.sigmoid(model(x)) > 0.5).float()
            # compute the Dice score
            dice_score += dice_coeff(mask_pred, y, reduce_batch_first=False)
            
            # preds = torch.sigmoid()
            # preds = (preds > 0.5).float()

            # num_correct += (preds == y).sum().item()
            # num_pixels += torch.numel(preds)

    print(
        dice_score / max(len(loader), 1)
    )

    
    model.train()
    
    # return num_correct / num_pixels
    
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask

    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)