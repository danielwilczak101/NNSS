from sklearn.model_selection import train_test_split
from statistics import mean
from math import sqrt

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os.path
import h5py


from functions import to_coefs, to_labels, normalize, unnormalize, renormalize, get_poly_labels

print("Loading")
hf = h5py.File('dataset/dataset_raw.h5', 'r')


images = hf['images']
labels = hf['spectra']

train_images, test_images, train_labels, test_labels = train_test_split(
    images,
    labels,
    shuffle=False,
    test_size=0.2,
)

train_images, images_mean, images_deviation = normalize(train_images)
train_labels, labels_mean, labels_deviation = normalize(train_labels)


poly_labels = {}


degree = 6
#train_poly, test_poly = get_poly_labels(degree)

# models = [tf.keras.models.load_model(f"models/poly_models/{index}-0.h5")
# for index in range(3, 7)]

is_poly_augmented = False
# f"models/{'no_' * (1 - is_poly_augmented)}poly_models/14-{index}.h5")
models = [tf.keras.models.load_model(f"models/2-{index}.h5")
          for index in range(1)]


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

image_number = 2000

plt.plot(unnormalize(test_labels[image_number], labels_mean, labels_deviation), label="Actual")

model_labels = np.vstack([
    unnormalize(
        model(renormalize(
            test_images[image_number].reshape(1, 64, 64, 3),
            images_mean,
            images_deviation,
        ))[0],
        labels_mean,
        labels_deviation,
    )
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
