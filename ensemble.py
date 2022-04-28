"""
Creates random ensembled models using a range
specified for the metal oxide dataset.
"""

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from tqdm import tqdm
from PIL import Image

import keras
from keras.models import Sequential
from keras.layers import *

import tensorflow as tf
from tensorflow.keras.models import Model

dataset = np.load("dataset/metal_oxide.npz")

layers = [1,2,3,4,5]
neurons = [256,512,1028,2048]

model = Sequential()
model.add(Flatten())
model.add(BatchNormalization())

layers = random.choice(layers)
print(layers)
for _ in range(layers):
    neuron_selection = random.choice(neurons)
    print(neuron_selection)
    model.add(Dense(neuron_selection, activation='relu'))

model.add(Dense(220, activation='sigmoid'))

opt = tf.keras.optimizers.Adam(0.0001)
rmse = tf.keras.metrics.RootMeanSquaredError()
model.compile(loss='mse', optimizer=opt, metrics=["mae", rmse])

h1 = model.fit(
        dataset["training_images"][:100000],
        dataset["training_labels"][:100000],
        validation_data=(
         dataset["testing_images"],
         dataset["testing_labels"]),
        verbose=1,
        epochs=100,
        batch_size=100
    )

model.save("models/5.h5")