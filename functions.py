from numpy.polynomial import Polynomial

import urllib.request
import os


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


def download_dataset():
    if os.path.exists("dataset/dataset_raw.h5") == False:
        download("dataset_raw.h5", "https://bit.ly/34xI5LW")
        download("training_raw.txt", "https://bit.ly/3rpWZwP")
        download("test_raw.txt", "https://bit.ly/34xOneu")
