from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String

engine = create_engine('sqlite:///database.db', echo=True)

meta = MetaData()

students = Table(
    'parameters', meta,
    Column('id', Integer, primary_key=True),
    Column('name', String),
    Column('lastname', String),
)


query = students.select().where(students.c.id < 3)
conn = engine.connect()
result = conn.execute(query)

for row in result:
    print(row)
