from email.policy import default
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship
engine = create_engine('sqlite:///database.db', echo=True)
meta = MetaData()

parameters = Table(
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

epochs = Table(
    'epochs', meta,
    Column('id', Integer, primary_key=True),
    Column('parameters_id', Integer, ForeignKey('parameters.parameters_id')),
    Column('model_id', Integer),
    Column('mean_square_error', Float),
    Column('mean_absolute_error', Float),
    Column('root_mean_error', Float),
    Column('elapsed_time', Float)
)

meta.create_all(engine)
