
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from numpy.polynomial import Polynomial
from functions import to_labels, to_coefs

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import os.path


number_of_models = 10

print("Loading")
hf = h5py.File('dataset_raw.h5', 'r')

print("Grabbing")
images = hf['images'][:10]
labels = hf['spectra'][:10]

print("Splitting...")
train_images, test_images, train_labels, test_labels = train_test_split(
    images,
    labels,
    shuffle=False,
    test_size=0.1,

)

plt.plot(train_labels[1], label="Actual")


print("Data Augmentation.")
degree = 16
train_labels = np.vstack([to_coefs(label, degree) for label in train_labels])
test_labels = np.vstack([to_coefs(label, degree) for label in test_labels])


print("Using the models")
models = [tf.keras.models.load_model(str(index)+'.h5')
          for index in range(number_of_models)]


for index, model in enumerate(models):
    prediction = model.predict(train_images[1].reshape(1, 64, 64, 3))
    if index == 0:
        plt.plot(to_labels(train_labels[1]), label="Augmented Actual")
    plt.plot(to_labels(prediction[0]), label=f"Model #{index} Prediction")


plt.legend()
plt.show()
