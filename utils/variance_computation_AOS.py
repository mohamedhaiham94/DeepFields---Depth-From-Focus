
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


size = (1024,1024)  # For example, resizing images to 200x200

script_dir = os.path.dirname(os.path.realpath(__file__))

os.chdir(script_dir)

print(f"Current working directory: {os.getcwd()}")

def load_images_with_stats(rgb_folder, alpha_folder, size):
        
    rgb_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith('.png')],
                        key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    alpha_files = sorted([f for f in os.listdir(alpha_folder) if f.endswith('.png')],
                        key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    rgb_images = []
    alpha_images = []
    
    for rgb_file, alpha_file in zip(rgb_files, alpha_files):

        rgb_img = Image.open(os.path.join(rgb_folder, rgb_file)).convert('RGB')
        rgb_img = rgb_img.resize(size)  # Resize to the specified size
        rgb_array = np.array(rgb_img)
        
        alpha_img = Image.open(os.path.join(alpha_folder, alpha_file)).convert('L')
        alpha_img = alpha_img.resize(size)  # Resize to the specified size

        alpha_array = np.array(alpha_img)
        
        rgb_images.append(rgb_array)
        alpha_images.append(alpha_array)

    return rgb_images, alpha_images


def compute_spatial_variance(image, kernel_size=2):
    # Compute the mean of the neighborhood
    mean = cv2.blur(image, (kernel_size, kernel_size))
    
    # Compute the squared mean of the neighborhood
    squared_mean = cv2.blur(image ** 2, (kernel_size, kernel_size))

    # Variance = E[x^2] - (E[x])^2
    variance = squared_mean - mean ** 2

    return variance


def sliding_window_std(image, window_size=(2, 2)):
    height, width, channels = image.shape
    window_height, window_width = window_size
    
    # To store std dev for each window
    std_dev_image = np.full((height - window_height + 1, width - window_width + 1, channels), np.nan)
    
    for y in range(height - window_height + 1):
        for x in range(width - window_width + 1):
            # Extract the 2x2 block for each channel
            window = image[y:y + window_height, x:x + window_width, :]
            pixel_coords = [(y + dy, x + dx) for dy in range(window_height) for dx in range(window_width)]

            valid_pixels = ~np.isnan(window)  
            
            if np.count_nonzero(valid_pixels) == window_height * window_width * channels:
                for c in range(channels):
                    std_dev_image[y, x, c] = np.nanstd(window[:, :, c])
    
    return std_dev_image


def angular_STD(rgb_images,alpha_images):
        height, width, _ = rgb_images[0].shape
        
        min_variance_image = np.zeros((height, width))
        mean_variance_image = np.zeros((height, width))
        max_variance_image = np.zeros((height, width))

        # min_mean_image = np.zeros((height, width))
        # mean_mean_image = np.zeros((height, width))
        # max_mean_image = np.zeros((height, width))

        # min_CV_image = np.zeros((height, width))
        # mean_CV_image = np.zeros((height, width))
        # max_CV_image = np.zeros((height, width))
        
        
        
        for y in range(height):
            for x in range(width):

                pixel_values = np.array([img[y, x] for img, alpha in zip(rgb_images, alpha_images) if alpha[y, x] == 1]) /255

                if pixel_values.shape[0] == 0:
                    continue
                
                r_mean, g_mean, b_mean = np.mean(pixel_values, axis=0)
                

                ### angular STD / MEAN /CV ###
                
                r2, g2, b2 = 0, 0, 0
                
                for i in range(len(pixel_values)):
                    r2 += (pixel_values[i][0] - r_mean) ** 2
                    g2 += (pixel_values[i][1] - g_mean) ** 2
                    b2 += (pixel_values[i][2] - b_mean) ** 2
                
                r_variance1 = r2 / ((len(pixel_values))-1)
                g_variance1 = g2 / ((len(pixel_values))-1)
                b_variance1 = b2 / ((len(pixel_values))-1)
                
                r_variance2 = np.sqrt(r_variance1)
                g_variance2 = np.sqrt(g_variance1)
                b_variance2 = np.sqrt(b_variance1)
                
                r_variance = r_variance2 #1 if r_mean == 0 else r_variance2 / r_mean
                g_variance = g_variance2 #1 if g_mean == 0 else g_variance2 / g_mean
                b_variance = b_variance2 #1 if b_mean == 0 else b_variance2 / b_mean

                # r_CV = 1 if r_mean == 0 else r_variance2 / r_mean
                # g_CV = 1 if g_mean == 0 else g_variance2 / g_mean
                # b_CV = 1 if b_mean == 0 else b_variance2 / b_mean
                ### angular STD / MEAN /CV ###
                

                min_variance_image[y, x] = np.min([r_variance, g_variance, b_variance])
                mean_variance_image[y, x] = np.mean([r_variance, g_variance, b_variance])
                max_variance_image[y, x] = np.max([r_variance, g_variance, b_variance])

                # min_mean_image[y, x] = np.min([r_mean, g_mean, b_mean])
                # mean_mean_image[y, x] = np.mean([r_mean, g_mean, b_mean])
                # max_mean_image[y, x] = np.max([r_mean, g_mean, b_mean])

                # min_CV_image[y, x] = np.min([r_CV, g_CV, b_CV])
                # mean_CV_image[y, x] = np.mean([r_CV, g_CV, b_CV])
                # max_CV_image[y, x] = np.max([r_CV, g_CV, b_CV])
                
               
        return min_variance_image,mean_variance_image,max_variance_image


def angular_entropy(rgb_images,alpha_images):
        height, width, _ = rgb_images[0].shape
        
        
        min_entropy_image = np.zeros((height, width))
        mean_entropy_image = np.zeros((height, width))
        max_entropy_image = np.zeros((height, width))
        
       
        
        for y in range(height):
            for x in range(width):

                pixel_values = np.array([img[y, x] for img, alpha in zip(rgb_images, alpha_images) if alpha[y, x] == 1]) /255

                if pixel_values.shape[0] == 0:
                    continue
                
                

                ## angular entropy ###
                pixel_values = pixel_values * 255
                hist_r, _ = np.histogram(pixel_values[:, 0], bins=256, range=(0, 255), density=True)
                hist_g, _ = np.histogram(pixel_values[:, 1], bins=256, range=(0, 255), density=True)
                hist_b, _ = np.histogram(pixel_values[:, 2], bins=256, range=(0, 255), density=True)
                
                def compute_entropy(hist):
                    hist = hist[hist > 0]  
                    return -np.sum(hist * np.log(hist))

                entropy_r = compute_entropy(hist_r)
                entropy_g = compute_entropy(hist_g)
                entropy_b = compute_entropy(hist_b)
                ## angular entropy ###

                min_entropy_image[y, x] = np.min([entropy_r, entropy_g, entropy_b])
                mean_entropy_image[y, x] = np.mean([entropy_r, entropy_g, entropy_b])
                max_entropy_image[y, x] = np.max([entropy_r, entropy_g, entropy_b])
                
        return min_entropy_image,mean_entropy_image,max_entropy_image


def spatial_STD(rgb_images,alpha_images):
        height, width, _ = rgb_images[0].shape
        
        
        
        min_avgSTD_image = np.zeros((height, width))
        mean_avgSTD_image = np.zeros((height, width))
        max_avgSTD_image = np.zeros((height, width))
        
        for y in range(height):
            for x in range(width):

                pixel_values = np.array([img[y, x] for img, alpha in zip(rgb_images, alpha_images) if alpha[y, x] == 1]) /255

                if pixel_values.shape[0] == 0:
                    continue
                                
                ## angular local STD ###
                actual_positions = np.array([i for i, (img, alpha) in enumerate(zip(rgb_images, alpha_images)) if alpha[y, x] == 1])
                pixel_values_std = np.array(pixel_values.tolist())
                actual_positions_std = np.array(actual_positions.tolist())
                
                grid_image = np.full((10, 10, 3), np.nan, dtype=np.float32)
                
                for i, pos in enumerate(actual_positions_std):
                    row = pos // 10  
                    col = pos % 10   
                    grid_image[row, col] = pixel_values_std[i]
                    
                std_dev_window_image = sliding_window_std(grid_image)
                
                
                min_std_dev_values = np.nanmin(std_dev_window_image, axis=-1)
                mean_std_dev_values = np.nanmean(std_dev_window_image, axis=-1)
                max_std_dev_values = np.nanmax(std_dev_window_image, axis=-1) 
                
                min_avgSTD_image[y, x] = np.nanmean(min_std_dev_values)
                mean_avgSTD_image[y, x] = np.nanmean(mean_std_dev_values) 
                max_avgSTD_image[y, x] = np.nanmean(max_std_dev_values)
                ## angular local STD ###

                
                
        return min_avgSTD_image,mean_avgSTD_image,max_avgSTD_image 




def create_or_clear_folder(folder_path):
    if os.path.exists(folder_path):
       
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
       
        os.makedirs(folder_path)



min_folder = r'./min_results'  # Folder to save min images
mean_folder = r'./mean_results' # Folder to save mean images
max_folder = r'./max_results'  # Folder to save max images

create_or_clear_folder(min_folder)
create_or_clear_folder(mean_folder)
create_or_clear_folder(max_folder)

alpha_folders = [d for d in os.listdir('./variance') if d.startswith('alpha')]


# for i in range(len(alpha_folders)):
    
#     rbg_path = r"./variance/shifted_images_{}".format(i)
#     alpha_path = r"./variance/alpha_{}".format(i)

#     rgb_images, alpha_images = load_images_with_stats(rbg_path, alpha_path, size=size)


for i in range(len(alpha_folders)):

    rbg_path = r"./variance/shifted_images_{}".format(i)
    alpha_path = r"./variance/alpha_{}".format(i)

    rgb_images, alpha_images = load_images_with_stats(rbg_path, alpha_path, size=size)

    print("Processing angular STD image: ",i)
    
    min_variance_image,mean_variance_image,max_variance_image = angular_STD(rgb_images,alpha_images)

    #Angular STD Images
    min_image = Image.fromarray(min_variance_image)
    min_image.save(r"./min_results/min_variance_image_{}.tiff".format(i+1))
    
    mean_image = Image.fromarray(mean_variance_image)
    mean_image.save(r"./mean_results/mean_variance_image_{}.tiff".format(i+1))
    
    max_image = Image.fromarray(max_variance_image)
    max_image.save(r"./max_results/max_variance_image_{}.tiff".format(i+1))


    # #Angular mean images
    # min_image = Image.fromarray(min_mean_image)
    # min_image.save(r"./min_results/min_mean_image_{}.tiff".format(i+1))
    
    # mean_image = Image.fromarray(mean_mean_image)
    # mean_image.save(r"./mean_results/mean_mean_image_{}.tiff".format(i+1))

    # max_image = Image.fromarray(max_mean_image)
    # max_image.save(r"./max_results/max_mean_image_{}.tiff".format(i+1))


    # #Angular CV images
    # min_image = Image.fromarray(min_CV_image)
    # min_image.save(r"./min_results/min_CV_image_{}.tiff".format(i+1))
    
    # mean_image = Image.fromarray(mean_CV_image)
    # mean_image.save(r"./mean_results/mean_CV_image_{}.tiff".format(i+1))

    # max_image = Image.fromarray(max_CV_image)
    # max_image.save(r"./max_results/max_CV_image_{}.tiff".format(i+1))


    # #Spatial Variance Images
    # min_spatial_variance=compute_spatial_variance(min_mean_image, 5)
    # pil_image = Image.fromarray(min_spatial_variance)
    # pil_image.save(r"./min_results/min_spatial_variance_image_{}.tiff".format(i+1))

    # mean_spatial_varianve=compute_spatial_variance(mean_mean_image, 5)
    # pil_image = Image.fromarray(mean_spatial_varianve)
    # pil_image.save(r"./mean_results/mean_spatial_variance_image_{}.tiff".format(i+1))

    # max_spatial_varianve=compute_spatial_variance(max_mean_image, 5)
    # pil_image = Image.fromarray(max_spatial_varianve)
    # pil_image.save(r"./max_results/max_spatial_variance_image_{}.tiff".format(i+1)) 


for i in range(len(alpha_folders)):

    rbg_path = r"./variance/shifted_images_{}".format(i)
    alpha_path = r"./variance/alpha_{}".format(i)

    rgb_images, alpha_images = load_images_with_stats(rbg_path, alpha_path, size=size)

    print("Processing angular entropy image: ",i)      
    
    min_entropy_image,mean_entropy_image,max_entropy_image = angular_entropy(rgb_images,alpha_images)


    
    # Angular Entropy Images
    min_image = Image.fromarray(min_entropy_image)
    min_image.save(r"./min_results/min_entropy_image_{}.tiff".format(i+1))
    
    mean_image = Image.fromarray(mean_entropy_image)
    mean_image.save(r"./mean_results/mean_entropy_image_{}.tiff".format(i+1))

    max_image = Image.fromarray(max_entropy_image)
    max_image.save(r"./max_results/max_entropy_image_{}.tiff".format(i+1))


for i in range(len(alpha_folders)):

    rbg_path = r"./variance/shifted_images_{}".format(i)
    alpha_path = r"./variance/alpha_{}".format(i)

    rgb_images, alpha_images = load_images_with_stats(rbg_path, alpha_path, size=size)

    print("Processing spatial STD image: ",i)    
    
    min_avgSTD_image,mean_avgSTD_image,max_avgSTD_image = spatial_STD(rgb_images,alpha_images)

    #Spatial SDT on Angular Patches Images
    min_image = Image.fromarray(min_avgSTD_image)
    min_image.save(r"./min_results/min_avgSTD_image_{}.tiff".format(i+1))
    
    mean_image = Image.fromarray(mean_avgSTD_image)
    mean_image.save(r"./mean_results/mean_avgSTD_image_{}.tiff".format(i+1))

    max_image = Image.fromarray(max_avgSTD_image)
    max_image.save(r"./max_results/max_avgSTD_image_{}.tiff".format(i+1))
    
    