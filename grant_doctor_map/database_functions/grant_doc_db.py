import pandas as pd
import sqlalchemy

from grants import grants_reader
from npi import npi_reader


def db():
    engine = sqlalchemy.create_engine('sqlite:///data/grant_npi.db')
    conn = engine.connect()
    return conn


def grants_csv_to_db():
    df = grants_reader.read_grants_year(22)
    df.to_sql('grants',
              db(),
              if_exists='append',
              index=False)
    
def npi_csv_to_db(csv_path: str):
    df = npi_reader.read(csv_path)
    df.to_sql('npi',
              db(),
              if_exists='append',
              index=False)


if __name__ == '__main__':
    grants_csv_to_db()
    npi_csv_to_db('data/npidata_pfile_20240205-20240211.csv')