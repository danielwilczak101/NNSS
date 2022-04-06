#!/usr/bin/env python3.8.9
from sklearn.model_selection import train_test_split
from functions import generate_combinations

import itertools
import h5py

combinations = generate_combinations()

print("Splitting...")
hf = h5py.File('dataset/dataset_raw.h5', 'r')
images = hf['images']
labels = hf['spectra']

for id, combination in enumerate(combinations):

    current_poly = combination[0]
    networks = combination[1]
    layers = combination[2]
    neurons = combination[3]
    bias = combination[4]
    epochs = combination[5]
    current_training_split = combination[6]

    old_training_split = None
    old_poly = None

    if old_training_split is not current_training_split:
        print("Changing split data")

    if old_poly is not current_poly:
        print("Changing poly count")


def model(id: int, polys: int, networks: int, layers: int, bias: bool, epochs: int, training_split: float):
    pass
