"""
Creates a base model that will be used to compare
to the ensembled model.
"""

import os
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

np.random.seed(1)

model = Sequential()
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(2048, activation='relu',))
model.add(Dense(2048, activation='relu'))
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

model.save("base.h5")