# 1. SQL Database --> Create blocks out of last names
# 2. Use group by to create blocks in the SQL DB. Get IDs first, then loop thru them
# 3. For each last name, do the checks
# 4. Get pairs of dfs for 1 name at a time
# 5. 

import pandas as pd
import numpy as np
import entity_resolution_model
import sqlite3
import model_features
import entity_resolution_model
def map():
    conn = sqlite3.connect('data/grant_npi.db')
    query = '''SELECT DISTINCT last_name FROM grants;'''
    names = pd.read_sql(query,con=conn)
    query = '''SELECT * FROM grants GROUP BY last_name'''
    grants = pd.read_sql(query,con=conn)
    query = '''SELECT * FROM npi GROUP BY last_name'''
    npi = pd.read_sql(query,con=conn)
    for name in names['last_name']:
        temp_grants = grants.loc[grants['last_name']==name]
        temp_npi = npi.loc[npi['last_name']==name]
        df = model_features.features(temp_grants,temp_npi)
        model = entity_resolution_model.EntityResolutionModel(model_dir='data')
        out = model.predict(features=df)
        print()
        #.loc[is_match] 
        # df.to_sql
    series = 0#entity_resolution_model.predict(df)
    return series

if __name__ == "__main__":
    map()