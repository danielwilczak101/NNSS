from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from numpy.polynomial import Polynomial
from functions import download, to_coefs
import time

import urllib.request
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import os.path


print("Downloading...")
if os.path.exists("dataset/dataset_raw.h5") == False:
    download("dataset_raw.h5", "https://bit.ly/34xI5LW")
    download("training_raw.txt", "https://bit.ly/3rpWZwP")
    download("test_raw.txt", "https://bit.ly/34xOneu")

hf = h5py.File('dataset/dataset_raw.h5', 'r')

images = hf['images']
labels = hf['spectra']

print("Splitting...")
train_images, test_images, train_labels, test_labels = train_test_split(
    images,
    labels,
    shuffle=False,
    test_size=0.1
)

print("Data Augmentation.")
degree = 16
train_labels = np.vstack([to_coefs(label, degree) for label in train_labels])
test_labels = np.vstack([to_coefs(label, degree) for label in test_labels])

overall_history = []

print("Start timing:")
start = time.time()

print("Creating/Training models...")
number_of_models = 2
for index in range(number_of_models):
    model = models.Sequential()

    # Dense
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(17))

    # Complile and fit
    model.compile(optimizer=tf.optimizers.Adam(0.001),
                  loss='mse', metrics=["mae", tf.keras.metrics.RootMeanSquaredError()])
    history = model.fit(train_images, train_labels, epochs=2)
    overall_history.append([history.history])
    # Save
    model.save(str(index) + '.h5')

end = time.time()
print(f"Elapsed time: {end - start}")

print("Overall history:")
print(overall_history)
