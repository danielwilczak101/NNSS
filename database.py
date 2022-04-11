from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Boolean, ForeignKey


class Database:

    def __init__(self):
        engine = create_engine('sqlite:///database.db')
        meta = MetaData()
        self.table_parameters = Table(
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

        self.table_epochs = Table(
            'epochs', meta,
            Column('id', Integer, primary_key=True),
            Column('parameters_id', Integer, ForeignKey(
                'parameters.parameters_id')),
            Column('model_id', Integer),
            Column('epoch', Integer),
            Column('mean_square_error', Float),
            Column('mean_absolute_error', Float),
            Column('root_mean_squared_error', Float)
        )

        self.table_elapsed_time = Table(
            'elapsed_time', meta,
            Column('id', Integer, primary_key=True),
            Column('parameters_id', Integer,
                   ForeignKey('parameters.parameters_id')),
            Column('model_id', Integer,
                   ForeignKey('epochs.model_id')),
            Column('elapsed_time', Integer)
        )

        meta.create_all(engine)
        self.conn = engine.connect()

    def add_elapsed_time(self, parameter_id, model_id, elapsed_time):
        self.conn.execute(self.table_elapsed_time.insert().values(
            parameters_id=parameter_id,
            model_id=model_id,
            elapsed_time=elapsed_time
        ))

    def save_parameters(self, split, poly_aug, poly_degree, network, layers, neurons, epochs):
        """Creates the parameters id that the epoch data refrence to the overal ensembled model."""

        return (self.conn.execute(self.table_parameters.insert().values(
            split=split,
            poly_aug=poly_aug,
            poly_degree=poly_degree,
            network=network,
            layers=layers,
            neurons=neurons,
            epochs=epochs
        )).inserted_primary_key)[0]

    def save_epoch(self, data, epochs, parameter_id, model_id):
        """Saves the data the model creates after training a single model"""

        epoch_data = []

        for epoch in range(epochs):
            for key in data:
                epoch_data.append(data[key][epoch])

            self.conn.execute(self.table_epochs.insert().values(
                parameters_id=parameter_id,
                model_id=model_id,
                epoch=epoch,
                mean_square_error=epoch_data[0],
                mean_absolute_error=epoch_data[1],
                root_mean_squared_error=epoch_data[2]))

            epoch_data = []
