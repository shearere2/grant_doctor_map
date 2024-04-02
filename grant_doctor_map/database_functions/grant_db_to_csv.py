import pandas as pd
import sqlite3

def create_connection(db_file):
    """Creates the connection to db_file

    Args:
        db_file (str): Database file name to retrieve data from

    Returns:
        conn: Returns the connection
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except conn.Error as e:
        print(e)

    return conn

def select_npi_data(conn) -> pd.DataFrame:
    """Query each row in the npi table

    Args:
        conn (_type_): Connection to SQLite

    Returns:
        pd.DataFrame: Dataframe filled with data retrieved from the database
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM npi")

    rows = cur.fetchall()

    df = pd.DataFrame()
    for row in rows:
        df = df.add(row)

    return df

        
def grant_db_to_df(individual) -> pd.DataFrame:
    """Retrieves data

    Args:
        individual: Data to get?

    Returns:
        pd.DataFrame: Data retrieved from database.
    """
    conn = create_connection('grant_npi.db')
    select_npi_data(conn)