from math import sqrt
from statistics import mean
import numpy as np 

def ensemble_label(ensemble, image):
    """Returns the average result from all of the models in the ensemble."""
    return sum(model(image[None, ...])[0].numpy() for model in ensemble) / len(ensemble)

def mse(ensemble, images, labels):
    """
    Returns the mean squared error given the ensemble, images, and labels.

    ensemble = [model, ...]
    images = [image, ...]
    labels = [label, ...]
    error = mse(ensemble, images, labels)
    """
    errors = (
        ((ensemble_label(ensemble, image) - label) ** 2).mean()
        for image, label in zip(images, labels)
    )
    return mean(errors)

def mae(ensemble, images, labels):
    """
    Returns the mean absolute error given the ensemble, images, and labels.

    ensemble = [model, ...]
    images = [image, ...]
    labels = [label, ...]
    error = mae(ensemble, images, labels)
    """
    errors = (
        np.abs(ensemble_label(ensemble, image) - label).mean()
        for image, label in zip(images, labels)
    )
    return mean(errors)

def rmse(ensemble, images, labels):
    """
    Returns the square root of the mean squared error given
    the ensemble, images, and labels.

    ensemble = [model, ...]
    images = [image, ...]
    labels = [label, ...]
    error = rmse(ensemble, images, labels)
    """
    return sqrt(mse(ensemble, images, labels))


def uq(ensemble, image):
    """
    Returns the mean and deviations in the ensemble output.

    ensemble = [model, ...]
    image = [data, ...]
    mean, deviation = uq(ensemble, image)
    plt.plot(mean)
    plt.fill_between(range(len(mean)), mean - deviation, mean + deviation)
    """
    mean = 0.0
    variance_sum = 0.0
    for i, model in enumerate(ensemble, start=1):
        prediction = model(image[None, ...])[0].numpy()
        error = prediction - mean
        variance_sum += (1 - 1 / i) * error ** 2
        mean += error / i
    return mean, np.sqrt(variance_sum / len(ensemble))