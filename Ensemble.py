from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from functions import to_coefs
from database import Database
from time import time

import tensorflow as tf
import numpy as np
import itertools
import h5py


class Ensemble:
    """Main class for all ensembled functionality."""

    def __init__(self, split=None, poly_aug=None, poly_degree=None, network=None, layers=None, neuron=None, epochs=None):
        self.combinations = list(itertools.product(*[
            split,
            poly_aug,
            poly_degree,
            network,
            layers,
            neuron,
            epochs,
        ]))

        # Variables used for triggering dataset changes.
        self.current_split_percentage = 0
        self.current_poly_degree = 0
        self.current_poly_aug = 0

        # Orginal unsplit data.
        hf = h5py.File('dataset/dataset_raw.h5', 'r')
        self.images = hf['images']
        self.labels = hf['spectra']

        # Current data that is used in the model.
        self.training_images, self.training_labels = None, None
        self.testing_images, self.testing_labels = None, None

        # A record is kept of the original split image so when polynomial
        # augmentation is applied it uses the original data and not the
        # already modified data.
        self.original_training_images, self.original_training_labels = None, None
        self.original_testing_images, self.original_testing_labels = None, None

        # All database functionality is through the database class.
        self.database = Database()

    def check_split(self, percentage: float, id: int):
        """Check to see if the split variable has changed. If it has
        change then reset the training data variables."""
        if self.current_split_percentage is not percentage:

            # Update the self.training and self.testing variables with new training split
            print("Splitting Dataset")
            self.training_images, self.testing_images, self.training_labels, self.testing_labels = train_test_split(
                self.images,
                self.labels,
                shuffle=False,
                test_size=percentage
            )
            if id == 0:
                # Set the orginal data variables.
                self.original_training_images = self.training_images
                self.original_training_labels = self.training_labels
                self.original_testing_images = self.testing_images
                self.original_testing_labels = self.testing_labels

            if self.current_poly_aug != 0:
                # Aplly polynomial augmentation if it needed.
                self.apply_polynomial_augmentation(self.current_poly_degree)

            # Update the trigger value
            self.current_split_percentage = percentage

    def check_poly_aug(self, aug: int):
        """Check to see if the polynomial augmentation trigger has 
        changed. If it has changed then take the orginal data and apply
        polynomial augmentation to it and set that to the current data."""

        if self.current_poly_aug is not aug:
            # If the augmentagtion trigger is swapped then apply
            # poolynomial augmentation useing the degree variable.
            self.apply_polynomial_augmentation(self.current_poly_degree)
            # Update the trigger value
            self.current_poly_aug = aug

    def check_poly_degree(self, degree: int, poly_aug: bool):
        """Check to see if the polynomial degree has changed.  If it has
        changed then take the orginal data and apply polynomial augmentation
        to it and set that to the current data."""

        if poly_aug != 0:
            if self.current_poly_degree is not degree:
                # Update the self augmented data with the new degree
                self.apply_polynomial_augmentation(degree)
                # Update the trigger value
                self.current_poly_degree = degree

    def apply_polynomial_augmentation(self, degree):
        """Apply polynomial augmentation to the label data."""

        print(f"Applying polynomial Augmentation with degree: {degree}")
        self.training_labels = np.vstack(
            [to_coefs(label, degree) for label in self.original_training_labels])
        self.testing_labels = np.vstack(
            [to_coefs(label, degree) for label in self.original_testing_labels])

        print(self.training_labels[0])

    def model(
            self, parameter_id: int,
            poly_aug: bool, poly_degree: int,
            networks: int, layer_count: int, neurons: int, epochs: int):

        for model_id in range(networks):

            print(f"Currently Trianing: {parameter_id} - {model_id}")
            # Start log timer
            start = time()

            model = models.Sequential()
            model.add(layers.Flatten())

            # Generate the models layers
            for _ in range(layer_count):
                model.add(layers.Dense(neurons, activation='relu'))
            if poly_aug == 0:
                # For non poly augmented data
                model.add(layers.Dense(220))

            if poly_aug:
                # Poly augmented data. The addition of one is to componsate for
                # the a in a + a1x + a2x^2 + an^n
                model.add(layers.Dense(poly_degree+1))

            # Complile and fit
            model.compile(
                optimizer=tf.optimizers.Adam(0.001),
                loss='mse',
                metrics=[
                    "mae",
                    tf.keras.metrics.RootMeanSquaredError()
                ]
            )

            # Fit the data and grab the data it produces during training
            data = model.fit(self.training_images,
                             self.training_labels, epochs=epochs)

            # Time it took for the model to train.
            elapse_time = time() - start

            # Save the model
            model.save(f"models/{parameter_id}-{model_id}.h5")

            # Add the elapsed time to the data given by the model.
            self.database.add_elapsed_time(parameter_id, model_id, elapse_time)

            # Save the models epoch data
            self.database.save_epoch(
                data.history, epochs, parameter_id, model_id)

    def run(self, limit):
        """Runs the entire ensembled research based on preset combination parameters."""
        print(f"Number of combinations: {len(self.combinations)}")
        for id, combination in enumerate(self.combinations):
            if id < limit:
                print(f"Current combination: {combination}")
                split, poly_aug, poly_degree, networks, layers, neurons, epochs = combination

                if id == 0:
                    # For the first iteration set the poly augmentation varibales
                    # and degree since these are needed for the check split.
                    self.current_poly_aug = poly_aug
                    self.current_poly_degree = poly_degree

                # Checks are used to see if any augmentation variable have changed.
                self.check_split(split, id)
                self.check_poly_aug(poly_aug)
                self.check_poly_degree(poly_degree, poly_aug)

                # Save the combination to the database and recieve
                # and return an id for saving epoch data.
                parameter_id = self.database.save_parameters(split, poly_aug,
                                                             poly_degree, networks,
                                                             layers, neurons, epochs)

                self.model(parameter_id, poly_aug, poly_degree,
                           networks, layers, neurons, epochs)


"""ensemble = Ensemble(
    split=[.2],
    poly_aug=[1],
    poly_degree=[12, 18],
    network=[1, 3, 5],
    layers=[1, 3],
    neuron=[512, 1028],
    epochs=[2]
)"""


ensemble = Ensemble(
    split=[.2],
    poly_aug=[0],
    poly_degree=[12, 18],
    network=[5],
    layers=[3],
    neuron=[2056],
    epochs=[3]
)

ensemble.run(2)
