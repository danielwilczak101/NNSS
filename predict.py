from functions import normalize, unnormalize, renormalize
from statistics import mean
from Dataset import Dataset

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

data = Dataset()
data.check_split(.2, 0)

train_images, images_mean, images_deviation = normalize(
    data.original_training_images)
train_labels, labels_mean, labels_deviation = normalize(
    data.original_training_labels)

models = [tf.keras.models.load_model(
    f"models/2-{index}.h5") for index in range(1)]

# Image that will be used for everything
image_number = 10

# I NEED TO FIGURE THIS OUT.
model_labels = np.vstack([
    unnormalize(
        model(train_images[image_number].reshape(1, 64, 64, 3))[0],
        labels_mean,
        labels_deviation,
    )
    for model in models
])


# Compute mean and deviation of the labels.
mean = np.mean(model_labels, axis=0)
deviation = np.std(model_labels, axis=0)


# Graph the actual label
plt.plot(unnormalize(train_labels[image_number],
         labels_mean, labels_deviation), label="Actual")

# Graphing all models lines
for index, model in enumerate(models):
    plt.plot(model_labels[index], label=f"Model #{index} Prediction")

# Graph the variance.
plt.fill_between(range(220), mean - deviation, mean +
                 deviation, alpha=0.5, label="UQ")
plt.legend()
plt.show()
