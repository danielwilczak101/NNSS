from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from code.functions import download, to_coefs
import time

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


def create_model(
    ensemble_size: int,
    layers: int,
    neurons: int,
    degree: int,
    use_bias: bool,
    epochs: int,
):
    train_poly, test_poly = get_poly_labels(degree)
    models = []
    histories = []
    for i in range(ensemble_size):
        # Create the model.
        model = models.Sequential()
        model.add(layers.Flatten())
        for _ in range(layers):
            model.add(layers.Dense(neurons, activation="relu"))
        model.add(layers.Dense(degree + 1))
        # Setup the optimizer, loss, and metrics.
        model.compile(
            optimizer=tf.optimizers.Adam(0.001),
            loss='mse',
            metrics=[
                "mae",
                tf.keras.metrics.RootMeanSquaredError()
            ]
        )
        # Train the model and save the results.
        history = model.fit(train_images, train_poly, epochs=epochs)
        histories.append(history.history)
        model.save(f"model_{i}.h5")
        models.append(model)
    return models, histories
