
from tensorflow.keras import layers, models
from Database import Database
from Dataset import Dataset
from time import time

import tensorflow as tf
import numpy as np
import itertools


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

        # All database functionality is through the database class.
        self.database = Database()
        self.dataset = Dataset()

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
            else:
                # Poly augmented data. The addition of one is to componsate for
                # the a in a + a1x + a2x^2 + an^n
                model.add(layers.Dense(poly_degree + 1))

            # Complile and fit
            model.compile(
                optimizer=tf.optimizers.Adam(0.001),
                loss='mse',
                metrics=[
                    "mae",
                    tf.keras.metrics.RootMeanSquaredError(),
                ]
            )

            # Fit the data and grab the data it produces during training
            data = model.fit(self.dataset.training_images,
                             self.dataset.training_labels, epochs=epochs)

            # Time it took for the model to train.
            elapse_time = time() - start

            # Save the model
            model.save(f"models/{parameter_id}-{model_id}.h5")

            # Add the elapsed time to the data given by the model.
            self.database.add_elapsed_time(parameter_id, model_id, elapse_time)

            # Save the models epoch data
            self.database.save_epoch(
                data.history, epochs, parameter_id, model_id)

    def run(self):
        """Runs the entire ensembled research based on preset
        combination parameters."""

        print(f"Number of combinations: {len(self.combinations)}")

        for id, combination in enumerate(self.combinations):
            print(f"Current combination: {combination}")
            split, poly_aug, poly_degree, networks, layers, neurons, epochs = combination

            if id == 0:
                # For the first iteration set the poly augmentation varibales
                # and degree since these are needed for the check split.
                self.dataset.current_poly_aug = poly_aug
                self.dataset.current_poly_degree = poly_degree

            # Checks are used to see if any augmentation variable have changed.
            self.dataset.check_split(split, id)
            self.dataset.check_poly_aug(poly_aug)
            self.dataset.check_poly_degree(poly_degree, poly_aug)

            # Save the combination to the database and recieve
            # and return an id for saving epoch data.
            parameter_id = self.database.save_parameters(split, poly_aug,
                                                         poly_degree, networks,
                                                         layers, neurons, epochs)

            self.model(parameter_id, poly_aug, poly_degree,
                       networks, layers, neurons, epochs)
