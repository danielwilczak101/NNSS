from time import time
from database import Database

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

        # Used as trigger for when to change the dataset
        self.current_split = None
        self.current_poly = None
        self.current_poly_aug = None

        self.images = None
        self.labels = None

        self.training_images, self.training_labels = None, None
        self.testing_images, self.testing_labels = None, None

        self.database = Database()

    def check_split(self, percentage: float):
        """Check to see if the split variable has changed. If it has
        change then reset the training data variables."""
        if self.current_split is not percentage:

            # Update the self.training and self.testing variables with new training split
            print(f"Updating split % to {percentage} & changed variables")

            # Update the trigger value
            self.current_split = percentage

    def check_poly_aug(self, aug: int):
        if self.current_poly_aug is not aug:
            # Update the trigger for whether the data should be augmented or not.
            print("Changed Poly Augmentation")

            # Update the trigger value
            self.current_poly_aug = aug

    def check_poly_degree(self, degree: int, poly_aug: bool):
        if poly_aug != 0:
            if self.current_poly is not degree:

                # Update the self augmented data with the new degree
                print(f"Updating poly augmentation to - Degree of: {degree}")

                # Update the trigger value
                self.current_poly = degree

    def model(
            self, parameter_id: int,
            poly_aug: bool, poly_degree: int,
            networks: int, layers: int, neurons: int, epochs: int,
            images=None, labels=None):

        # Save the parameters to the database

        for index in range(networks):
            # used for identifying what model this is.
            model_id = f"{parameter_id}-{index}"

            # Start log timer
            start = time()

            model = [model_id]
            model.append(["Flatten"])

            # Generate the models layers
            for _ in range(layers):
                model.append([f"Dense Neurons:{neurons}"])

            if poly_aug == 0:
                # For non poly augmented data
                model.append([f"last layer Dense: {220}"])

            if poly_aug == 1:
                # Poly augmented data
                model.append([f"last layer Dense: {poly_degree} "])

            # Time it took for the model to train
            elapse_time = time() - start
            model.append(elapse_time)
            # The model produces the data below for each model.

            # Save the model
            open("models/"+model_id+".txt", "x")

            # Save the epoch information to the epoch table
            data = [{
                'loss': [12.675312042236328]*epochs,
                'mae': [1.168357491493225]*epochs,
                'root_mean_squared_error': [3.5602402687072754]*epochs,
                'elapsed_time': [elapse_time]*epochs
            }]

            self.database.save_epoch(data, epochs, parameter_id, index)

    def run(self):
        """Runs the entire ensembled research based on preset parameters."""
        for id, combination in enumerate(self.combinations):
            if id < 2:
                split, poly_aug, poly_degree, networks, layers, neurons, epochs = combination

                self.check_split(split)
                self.check_poly_aug(poly_aug)
                self.check_poly_degree(poly_degree, poly_aug)

                parameter_id = self.database.save_parameters(split, poly_aug,
                                                             poly_degree, networks,
                                                             layers, neurons, epochs)

                self.model(parameter_id, poly_aug, poly_degree,
                           networks, layers, neurons, epochs)


ensemble = Ensemble(
    split=[.2],
    poly_aug=[0, 1],
    poly_degree=[5, 8, 11],
    network=[1, 5, 10],
    layers=[1, 2, 3],
    neuron=[256, 512, 1028],
    epochs=[2, 4, 6]
)

ensemble.run()
