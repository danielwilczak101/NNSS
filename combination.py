import h5py
import itertools
from functions import generate_combinations
from sklearn.model_selection import train_test_split

print("Downloading files")
combinations = generate_combinations()

print("Splitting...")
hf = h5py.File('dataset/dataset_raw.h5', 'r')
images = hf['images']
labels = hf['spectra']

old_training_split = None
old_poly = None

for id, combination in enumerate(combinations):

    current_poly = combination[0]
    networks = combination[1]
    layers = combination[2]
    neurons = combination[3]
    bias = combination[4]
    epochs = combination[5]
    current_training_split = combination[6]

    if old_training_split is not current_training_split:
        print(
            f"Changing split data old:{old_training_split} new:{combination[6]}")
        old_training_split = current_training_split

    if old_poly is not current_poly:
        print(f"Changing poly count old:{old_poly} new:{combination[0]}")
        old_poly = current_poly


def model(id: int, polys: int, networks: int, layers: int, bias: bool, epochs: int, training_split: float):
    pass
