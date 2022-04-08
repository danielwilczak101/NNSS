from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Boolean, ForeignKey
from math import comb
from time import time
from numpy import poly

import itertools

engine = create_engine('sqlite:///database.db')
meta = MetaData()
table_parameters = Table(
    'parameters', meta,
    Column('parameters_id', Integer, primary_key=True),
    Column('split', Float),
    Column('poly_aug', Boolean),
    Column('poly_degree', Integer),
    Column('network', Integer),
    Column('layers', Integer),
    Column('neurons', Integer),
    Column('epochs', Integer)
)

table_epochs = Table(
    'epochs', meta,
    Column('id', Integer, primary_key=True),
    Column('parameters_id', Integer, ForeignKey('parameters.parameters_id')),
    Column('model_id', Integer),
    Column('mean_square_error', Float),
    Column('mean_absolute_error', Float),
    Column('root_mean_squared_error', Float),
    Column('elapsed_time', Float)
)
meta.create_all(engine)
conn = engine.connect()


class Ensemble:

    def __init__(self):
        self.split = [.2]
        self.poly_aug = [0, 1]
        self.poly_degree = [5, 8, 11]
        self.network = [1, 5, 10]
        self.layers = [1, 2, 3]
        self.neuron = [256, 512, 1028]
        self.epochs = [2, 4, 6]

        # Used for identifying the model
        self.id = 0

        # Used as trigger for when to change the dataset
        self.current_split = None
        self.current_poly = None
        self.current_poly_aug = None

        self.combinations = list(itertools.product(*[
            self.split,
            self.poly_aug,
            self.poly_degree,
            self.network,
            self.layers,
            self.neuron,
            self.epochs,
        ]))

        self.images = None
        self.labels = None

        self.training_images, self.training_labels = None, None
        self.testing_images, self.testing_labels = None, None

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

            self.save_epoch(data, epochs, parameter_id, index)

    def save_epoch(self, data, epochs, parameter_id, model_id):
        """Saves the data the model creates after training a single model"""
        data = data[0]
        epoch_data = []

        for element in range(epochs):
            for key in data:
                epoch_data.append(data[key][element])

            insert = table_epochs.insert().values(
                parameters_id=parameter_id,
                model_id=model_id,
                mean_square_error=epoch_data[0],
                mean_absolute_error=epoch_data[1],
                root_mean_squared_error=epoch_data[2],
                elapsed_time=epoch_data[3])

            conn.execute(insert)
            epoch_data = []

    def save_parameters(self, split, poly_aug, poly_degree, network, layers, neurons, epochs):
        """Saves the data the model creates after training a single model"""

        return (conn.execute(table_parameters.insert().values(
            split=split,
            poly_aug=poly_aug,
            poly_degree=poly_degree,
            network=network,
            layers=layers,
            neurons=neurons,
            epochs=epochs
        )).inserted_primary_key)[0]

    def run(self):
        """Runs the entire ensembled research based on preset parameters."""
        for id, combination in enumerate(self.combinations):
            if id < 2:
                split, poly_aug, poly_degree, networks, layers, neurons, epochs = combination

                self.check_split(split)
                self.check_poly_aug(poly_aug)
                self.check_poly_degree(poly_degree, poly_aug)

                parameter_id = self.save_parameters(split, poly_aug,
                                                    poly_degree, networks,
                                                    layers, neurons, epochs)

                self.model(parameter_id, poly_aug, poly_degree,
                           networks, layers, neurons, epochs)


ensemble = Ensemble()
ensemble.run()
