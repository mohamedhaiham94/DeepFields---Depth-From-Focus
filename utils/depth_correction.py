# import numpy as np
# import cv2 

# image = cv2.imread(r'd:\Research\Wild Fire - Project\Used Dataset\simulated\TD0_pose_0_depth.png', -1)


# # Replace all 0 values with 3500
# image[image == 0] = 35
# image = image - 30


# cv2.imwrite(r'd:\Research\Wild Fire - Project\Used Dataset\simulated\g.png', image)

import imageio.v3 as iio
import numpy as np
import os
import matplotlib.pyplot as plt
import OpenEXR
import Imath


os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

DIR = r'data/tt'
files = os.listdir(DIR)

def read_exr():
    exr_file = OpenEXR.InputFile(os.path.join(DIR, file))
        
    # Get the data window (dimensions) from the header
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # Get the channel names (typically 'R', 'G', 'B' for RGB images)
    channels = header['channels'].keys()
    
    # Read all channels into a dictionary
    full_image = {}
    for channel in channels:
        # Read channel as 32-bit float
        channel_data = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
        channel_array = np.frombuffer(channel_data, dtype=np.float32)
        channel_array = channel_array.reshape((height, width))
        full_image[channel] = channel_array
    
    exr_file.close()
    return full_image, (width, height)

for file in files:
    # Load the OpenEXR file
    image_data, dimensions = read_exr()
    image_data = image_data['G'].astype(np.float32)
    image_data = image_data.max() - image_data
    binary_depth = (image_data > 0).astype(np.uint8)

    
    # Not used anymore.
    # image_dataa = abs(image_data['G'].astype(np.float32) - 5) # the 5 here means the distance between the near clipping plane and the ground
    # By subtracting the max value from the depth map we say that pixel is on the plane and focus
    # and should have value of zero
    # depth_map = image_dataa.max() - image_dataa

    iio.imwrite(os.path.join(DIR, file.split('.')[0]+'.png'), np.array(binary_depth) * 255)