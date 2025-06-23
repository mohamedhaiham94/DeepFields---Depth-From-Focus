import torch
import torch.nn as nn
from model import VAE
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from torch.nn import functional as F
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from dataloader import PatchDataset
from torch.utils.data import DataLoader
import cv2
from sklearn.model_selection import GridSearchCV
import numpy as np
from PIL import Image
from tqdm import tqdm

def remove_small_components(mask, min_size=100):
    """
    Remove connected components smaller than `min_size`.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    cleaned_mask = np.zeros_like(mask)

    for i in range(1, num_labels):  # Skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_mask[labels == i] = 255

    return cleaned_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def expand_labels_to_image(labels, patch_grid_shape, patch_size=16):
    """
    Expand patch-level labels to full image resolution.
    
    Args:
        labels: 1D array of patch labels (length H*W)
        patch_grid_shape: tuple (rows, cols) of patch grid
        patch_size: size of each patch (default=16)

    Returns:
        image_mask: 2D array of shape (rows*patch_size, cols*patch_size)
    """
    rows, cols = patch_grid_shape
    label_grid = labels.reshape(rows, cols)

    # Repeat each label to form a block of patch_size x patch_size
    expanded = np.kron(label_grid, np.ones((patch_size, patch_size), dtype=int))
    return expanded

def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)

def test(vae, image_path, layer_name):
    """
    Compute pixel-wise likelihood of an image.
    Args:
        vae (nn.Module): Trained VAE model.
        image (numpy.ndarray): Grayscale image.
    Returns:
        numpy.ndarray: Pixel-wise likelihood map.
    """
    # Assuming patches is a NumPy array of shape (N, 1, 16, 16)
    image_path = image_path
    image = Image.open(image_path)

    # patch_dataset = PatchDataset(np.array(image), 4, 4)
    # patches = patch_dataset.get_patches()

    ###################

    patch_dataset = PatchDataset('data/Testing-scenes/Scene_2_dense_1_spacing_row_0.30_col_damage_2_seed_225/max_results/STD', 4, 4, i)
    patches = patch_dataset.get_patches()
    patches = np.transpose(patches, (0, 3, 1, 2))  # (N, 6, 4, 4)


    vae.eval()
    latents = []


    images = []
    with torch.no_grad():
        for patch in tqdm(patches, desc="Processing patches"):

            patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)
            loss, recon_mb, mu, logvar = vae.step(
                patch_tensor
            )    
            
            z = vae.reparameterize(mu, logvar)  # or just use mu
            latents.append(z.squeeze().cpu().numpy())

            # latents.append(recon_mb.squeeze().cpu().numpy())
    

    # # Step 1: Reshape to grid layout
    # patches = np.array(latents).reshape(256, 256, 4, 4)  # Now shape = (rows, cols, 4, 4)

    # # Step 2: Rearrange to form the full image
    # # We bring the 4x4 patches into one big 1024x1024 image
    # image_array = np.block([[patches[i, j] for j in range(256)] for i in range(256)])
    # cv2.imwrite('test.tiff', image_array)
    
    
    latents = np.stack(latents)

    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42, max_iter=300)

    # param_grid = {
    #     "covariance_type": ["spherical", "tied", "diag", "full"],
    # }
    # grid_search = GridSearchCV(
    #     GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
    # )
    # grid_search.fit(latents)
    # import pandas as pd

    # df = pd.DataFrame(grid_search.cv_results_)[
    #     ["param_covariance_type", "mean_test_score"]
    # ]
    # df["mean_test_score"] = -df["mean_test_score"]
    # df = df.rename(
    #     columns={
    #         "param_covariance_type": "Type of covariance",
    #         "mean_test_score": "BIC score",
    #     }
    # )
    # print(df.sort_values(by="BIC score").head())
    labels = gmm.fit_predict(latents)
    # Count values in each cluster
    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    min_cluster = min(cluster_counts.items(), key=lambda x: x[1])[0]

    print("Cluster counts:", cluster_counts)
    labels = np.array([0 if x != min_cluster else 1 for x in labels])

    patch_grid_shape = (256, 256)

    segmentation_mask = expand_labels_to_image(labels, patch_grid_shape, patch_size=4)
    # segmentation_mask = remove_small_components(segmentation_mask)
    # Assume segmentation_mask is (1024, 1024), binary (0 and 1)

    # from sklearn.manifold import TSNE

    # # reduce dimensionality to 2D, we consider a subset of data because TSNE
    # # is a slow algorithm
    # tsne_features = TSNE(n_components=2).fit_transform(latents[:2000])
    # fig = plt.figure(figsize=(10, 6))

    # plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels[:2000], marker='o',
    #             edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s = 10)
    # plt.grid(False)
    # plt.axis('off')
    # plt.colorbar()
    # plt.show()

    
    cv2.imwrite(f"data/Testing-scenes/Scene_2_dense_1_spacing_row_0.30_col_damage_2_seed_225/masks/{layer_name}.png", segmentation_mask.astype(np.uint8) * 255)
    
    print(segmentation_mask.shape)





for i in range(19, 101):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 128

    vae = VAE(latent_dim=latent_dim, depth = i).to(device)

    # Load the saved state dictionary
    vae.load_state_dict(torch.load(f"VAE/checkpoint/vae_scene_2_128_layer_{i}.pth"))
    image_path = f"data/Testing-scenes/Scene_2_dense_1_spacing_row_0.30_col_damage_2_seed_225/max_results/STD/max_variance_image_{i}.tiff"
    test(vae, image_path, i)
