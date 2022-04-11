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

images = hf['images'][:200]
labels = hf['spectra'][:200]

train_images, test_images, train_labels, test_labels = train_test_split(
    images,
    labels,
    shuffle=False,
    test_size=0.2,
)

poly_labels = {}


def get_poly_labels(degree: int):
    if degree not in poly_labels:
        poly_labels.clear()
        train_poly = np.vstack([
            to_coefs(label, degree)
            for label in train_labels
        ])
        test_poly = np.vstack([
            to_coefs(label, degree)
            for label in test_labels
        ])
        poly_labels[degree] = (train_poly, test_poly)
    return poly_labels[degree]


degree = 6
#train_poly, test_poly = get_poly_labels(degree)

# models = [tf.keras.models.load_model(f"models/poly_models/{index}-0.h5")
# for index in range(3, 7)]

is_poly_augmented = False

models = [tf.keras.models.load_model(f"models/{'no_' * (1 - is_poly_augmented)}poly_models/14-{index}.h5")
          for index in range(5)]


def as_labels(coefs_or_labels):
    if is_poly_augmented:
        return to_labels(coefs_or_labels)
    else:
        return coefs_or_labels


'''
for model in models:
    rmse = sqrt(mean(
        np.linalg.norm(
            as_labels(model(image.reshape(1, 64, 64, 3))[0]) - label
        ) ** 2
        for image, label in zip(test_images, test_labels)
    ))
    print(rmse)
'''

image_number = 5

plt.plot(test_labels[image_number], label="Actual")

model_labels = np.vstack([
    model(test_images[image_number].reshape(1, 64, 64, 3))[0]
    for model in models
])

for index, model in enumerate(models):
    plt.plot(model_labels[index], label=f"Model #{index} Prediction")

mean = np.mean(model_labels, axis=0)
deviation = np.std(model_labels, axis=0)

plt.fill_between(range(220), mean - deviation, mean +
                 deviation, alpha=0.5, label="UQ")

plt.legend()

plt.show()
