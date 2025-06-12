


'''

For precise resulte check notebooks/create_ground_truth.ipynb


'''


import os
import cv2
import numpy as np

def load_image_stack(folder_path, size=None):
    # Sorted load to ensure matching order
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.bmp'))])
    
    stack = []
    for file in image_files:
        img = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if size:
            img = cv2.resize(img, size)
        # Convert to binary: anything > 0 becomes 1
        binary = (img > 127).astype(np.uint8)
        stack.append(binary)
    
    return np.stack(stack, axis=0)  # Shape: (N, H, W)

# Paths to your two folders (each with 100 binary images)
folder1 = 'out/TopDown/STD_ENTROPY_3_CM'
folder2 = 'out/TopDown/STD_ENTROPY_3_CM_GR'

# Load both stacks
stack1 = load_image_stack(folder1)
stack2 = load_image_stack(folder2)

# Ensure both have the same shape
assert stack1.shape == stack2.shape, "Stacks do not match in shape"

# Total pixels
total_pixels = np.prod(stack1.shape)

# Compute metrics
accuracy = (stack1 == stack2).sum() / total_pixels
tp = np.logical_and(stack1 == 1, stack2 == 1).sum()
tn = np.logical_and(stack1 == 0, stack2 == 0).sum()
fp = np.logical_and(stack1 == 0, stack2 == 1).sum()
fn = np.logical_and(stack1 == 1, stack2 == 0).sum()

# Print results
print(f"Accuracy: {accuracy * 100:.4f}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
