from numpy.polynomial import Polynomial

import numpy as np
import urllib.request
import os


def download(file_name, url):
    urllib.request.urlretrieve(url, file_name)


def as_labels(coefs_or_labels, is_poly_augmented):
    if is_poly_augmented:
        return to_labels(coefs_or_labels)
    else:
        return coefs_or_labels


def get_poly_labels(degree: int, poly_labels, train_labels, test_labels):
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


def download_dataset():
    if os.path.exists("dataset/dataset_raw.h5") == False:
        download("dataset_raw.h5", "https://bit.ly/34xI5LW")
        download("training_raw.txt", "https://bit.ly/3rpWZwP")
        download("test_raw.txt", "https://bit.ly/34xOneu")


def normalize(x):
    """
    Normalizes the data. Returns the normalized data, the mean, and the deviation.

    training, mean, deviation = normalize(training)
    testing = renormalize(testing, mean, deviation)
    testing = unnormalize(testing, mean, deviation)
    """
    mean = np.mean(x, axis=0)
    deviation = np.std(x, axis=0)
    return renormalize(x, mean, deviation), mean, deviation


def renormalize(x, mean, deviation):
    """Normalizes the data using external mean and deviation."""
    return (x - mean) / deviation


def unnormalize(x, mean, deviation):
    """Unnormalizes the data using given mean and deviation."""
    return x * deviation + mean
