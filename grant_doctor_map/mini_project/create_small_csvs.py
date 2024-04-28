import sqlite3
import pandas as pd

def grant_csv():
    conn = sqlite3.connect('data/grant_npi.db')
    query = '''SELECT * FROM grant WHERE UPPER(SUBSTR(last_name, 1, 2)) = 'AB';'''
    names = pd.read_sql(query,con=conn)
    names.to_csv('data/grant_data_ab.csv')

def npi_csv():
    conn = sqlite3.connect('data/grant_npi.db')
    query = '''SELECT * FROM npi WHERE UPPER(SUBSTR(last_name, 1, 1)) = 'A';'''
    names = pd.read_sql(query,con=conn)
    names.to_csv('data/npi_data_ab.csv')

if __name__ == "__main__":
    grant_csv()
    npi_csv()