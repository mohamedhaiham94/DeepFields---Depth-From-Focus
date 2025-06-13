import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, sobel
from scipy.stats import rankdata
from skimage.transform import resize
import cv2 
def compute_pixel_features(image, window_size=5):
    pad = window_size // 2

    # 1. Raw STD value
    f_pixel_value = image

    # # 2. Local mean
    # f_local_mean = uniform_filter(image, size=window_size)
    # cv2.imwrite(r'D:\Research\3-Research(DeepFields)\Experiment\Depth\DeepFields - Depth From Focus\out\TopDown\f_local_mean.tiff', (f_local_mean).astype(np.float32))

    # # 3. Local standard deviation
    # f_local_sq_mean = uniform_filter(image**2, size=window_size)
    # f_local_std = np.sqrt(f_local_sq_mean - f_local_mean**2)
    # cv2.imwrite(r'D:\Research\3-Research(DeepFields)\Experiment\Depth\DeepFields - Depth From Focus\out\TopDown\f_local_std.tiff', (f_local_std).astype(np.float32))

    # # 4. Gradient magnitude using Sobel operator
    # grad_x = sobel(image, axis=0)
    # grad_y = sobel(image, axis=1)
    # f_gradient = np.hypot(grad_x, grad_y)
    # cv2.imwrite(r'D:\Research\3-Research(DeepFields)\Experiment\Depth\DeepFields - Depth From Focus\out\TopDown\f_gradient.tiff', (f_gradient).astype(np.float32))

    # # 5. Difference from global mean
    # global_mean = np.mean(image)
    # f_diff_global_mean = image - global_mean
    # cv2.imwrite(r'D:\Research\3-Research(DeepFields)\Experiment\Depth\DeepFields - Depth From Focus\out\TopDown\f_diff_global_mean.tiff', (f_diff_global_mean).astype(np.float32))

    # 6. Percentile rank (0 to 1)
    f_rank = rankdata(image.ravel(), method='average').reshape(image.shape)
    f_percentile = f_rank / f_rank.max()

    # Stack features into [H, W, 6]
    feature_stack = np.stack([
        f_pixel_value,
        # f_local_mean,
        # f_local_std,
        # f_gradient,
        # f_diff_global_mean,
        f_percentile
    ], axis=-1)

    return f_percentile

# === Usage Example ===

# Load your standard deviation image as a NumPy array (float32 recommended)
from PIL import Image

for i in range(1, 101):
    image_path = r"data/raw/TopDown/max/STD/max_variance_image_"+str(i)+".tiff"  # Replace with your path
    std_image = Image.open(image_path)
    std_array = np.array(std_image).astype(np.float32)

    # Compute features
    features = compute_pixel_features(std_array)
    cv2.imwrite(f'data/raw/TopDown/max/percentile_rank/percentile_rank_{str(i)}.tiff', (features).astype(np.float32))

# # Downsample for visualization (optional, for large images)
# visualization = resize(features, (256, 256, 6), anti_aliasing=True)

# # Plot the features
# titles = [
#     "1. Raw STD Value",
#     "2. Local Mean (5x5)",
#     "3. Local STD (5x5)",
#     "4. Gradient Magnitude (Sobel)",
#     "5. Diff from Global Mean",
#     "6. Percentile Rank"
# ]

# fig, axs = plt.subplots(2, 3, figsize=(18, 10))
# for i in range(6):
#     ax = axs[i // 3, i % 3]
#     im = ax.imshow(visualization[:, :, i], cmap='viridis')
#     ax.set_title(titles[i])
#     ax.axis('off')
#     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# plt.tight_layout()
# plt.show()
