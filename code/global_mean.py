from sklearn.model_selection import train_test_split
from statistics import mean
from math import sqrt

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os.path
import h5py


from functions import to_coefs, to_labels
#from use import models, test_images, test_labels

print("Loading")
hf = h5py.File('dataset/dataset_raw.h5', 'r')

plt.plot(np.mean(hf['spectra'], axis=0), label="global mean")
plt.title("Average of all spectrograms")

plt.show()
