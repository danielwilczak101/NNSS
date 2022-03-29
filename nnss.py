from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from numpy.polynomial import Polynomial

import urllib.request
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import os.path

def download(file_name, url): 
    urllib.request.urlretrieve(url, file_name)

def to_coefs(labels, degree):
    """Labels -> Coefficients."""
    return Polynomial.fit(range(220), labels, degree).coef * range(1, degree + 2) / (degree + 1) ** 2

def to_labels(coefs):
    """Coefficients -> Labels."""
    poly = to_poly(coefs)
    return poly(range(220))

def to_poly(coefs):
    """Coefficients -> Polynomial, for graphing."""
    degree = len(coefs) - 1
    return Polynomial(coefs / range(1, degree + 2) * (degree + 1) ** 2, domain=[0, 219], window=[-1, 1])


if os.path.exists("dataset_raw.h5") == False:
  download("dataset_raw.h5","https://bit.ly/34xI5LW")
  download("training_raw.txt","https://bit.ly/3rpWZwP")
  download("test_raw.txt","https://bit.ly/34xOneu")

images = hf['images']
labels = hf['spectra']

train_images, test_images, train_labels, test_labels = train_test_split(
    images,
    labels,
    shuffle=False,
    test_size = 0.1
    )

degree = 16
train_labels = np.vstack([to_coefs(label, degree) for label in train_labels])
test_labels = np.vstack([to_coefs(label, degree) for label in test_labels])

number_of_models = 2

for index in range(number_of_models):

  # Dense
  model.add(layers.Flatten(), activation='relu', input_shape=(64, 64, 3))
  model.add(layers.Dense(1024, activation='relu'))
  model.add(layers.Dense(17))

  # Complile and fit
  model.compile(optimizer=tf.optimizers.Adam(0.001), loss='mse', metrics=['mse', "mae", "mape"])
  model.fit(train_images, train_labels, epochs=5)

   # Save
  model.save(str(index) + '.h5')
  # Empty the model variable to conserve ram.