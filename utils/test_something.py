import os 
import cv2 
import numpy as np
import matplotlib.pylab as plt
import joblib


std_map = cv2.imread(r"out/TopDown/mona.tiff", cv2.IMREAD_UNCHANGED) 
gray= cv2.cvtColor(std_map,cv2.COLOR_BGR2GRAY)

# Applying SIFT detector
sift = cv2.SIFT_create()
kp = sift.detect(std_map, None)

# Marking the keypoint on the image using circles
img=cv2.drawKeypoints(gray ,
                      kp ,
                      std_map ,
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite(r'D:\Research\3-Research(DeepFields)\Experiment\Depth\DeepFields - Depth From Focus\out\TopDown\monaimage-with-keypoints.jpg', img)
dfg

from sklearn.mixture import GaussianMixture

X = std_map.flatten().reshape(-1, 1)
gmm = GaussianMixture(n_components=2, covariance_type='full').fit(X)
labels = gmm.predict(X)
labels_image = labels.reshape(std_map.shape)

labels_image = 1 - labels_image
print(labels_image)

cv2.imwrite(r'D:\Research\3-Research(DeepFields)\Experiment\Depth\DeepFields - Depth From Focus\out\TopDown\mona.tiff', (labels_image * std_map).astype(np.float32))


std_img = cv2.imread(r'D:\Research\3-Research(DeepFields)\Experiment\Depth\DeepFields - Depth From Focus\out\TopDown\mona.tiff', cv2.IMREAD_UNCHANGED) 


accuracy_curve = []


model = joblib.load(r"logs\TopDown\STD_0.03_CM_MLP\IMAGE_NUM_54\model.pkl") 


results = np.zeros((1024, 1024))


std_img = np.array(std_img)

H, W = std_img.shape
flat_std = std_img.flatten()
flat_avg = [std_img.mean()] * len(flat_std)

# Each pixel gets a feature vector: [std, std]
# feats = np.stack([flat_std, flat_avg], axis=1)  # shape: (H*W, 2)
feats = flat_std

preds = model.predict_proba(np.array(feats).reshape(-1, 1))
preds = (preds[:,1] >= 0.895).astype(int) # set threshold as 0.3

results = preds.reshape(H, W).astype(int)

cv2.imwrite(r"D:\Research\3-Research(DeepFields)\Experiment\Depth\DeepFields - Depth From Focus\out\TopDown\mona.png", results * 255)
#print(img_num)