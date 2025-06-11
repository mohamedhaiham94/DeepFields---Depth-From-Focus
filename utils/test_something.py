import os 
import pickle 
import numpy as np
import matplotlib.pylab as plt

file = open(r'data/spatial_data/training_data/input_vector_393_228_scene_1.pkl', 'rb')
input_vector = pickle.load(file)
file.close()

softmax = np.exp(input_vector[:, :, :, 0])/sum(np.exp(input_vector[:, :, :, 0]))

# plt.plot(softmax[:, 0, 0])
plt.plot(input_vector[:, 0, 0, 0] * softmax[:, 0, 0])
plt.show()
