
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


number_of_models = 2

hf = h5py.File('dataset_raw.h5', 'r')

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


print("Using the models")
models = [tf.keras.models.load_model(str(index)+'.h5')
          for index in range(number_of_models)]

for model in models:
    prediction = model.predict(train_images[0].reshape(1, 64, 64, 3))
    plt.plot(prediction[0], label="Model Prediction")
    plt.plot(train_labels[0], label="Actual")

plt.legend()
plt.show()
