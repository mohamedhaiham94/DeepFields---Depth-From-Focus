import os 
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re 

numbers = re.compile(r'(\d+)')
def numericalSort(value):
  parts = numbers.split(value)
  parts[1::2] = map(int, parts[1::2])
  return parts

def remove_iqr_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    return df[(df[column] <= lower_bound)], lower_bound


if __name__ == '__main__':
    
    DIR = r'data\raw\TopDown\max\STD'
    IMAGES = os.listdir(DIR)
    image_data = sorted(IMAGES,key=numericalSort)

    for i, image_path in enumerate(image_data):
        image = cv2.imread(os.path.join(DIR, image_path), cv2.IMREAD_UNCHANGED)
        
        flatten_image = image.flatten()
        df = pd.DataFrame()
        df['values'] = flatten_image
        
        _, lower_threhold = remove_iqr_outliers(df, 'values')

        ret, thresh_img = cv2.threshold(image, lower_threhold, 255, cv2.THRESH_BINARY)

        cv2.imwrite(f'out/manual_depth/{i+1}.png', thresh_img)


    print("Done")