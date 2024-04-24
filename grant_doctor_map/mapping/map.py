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
    model = entity_resolution_model.EntityResolutionModel(model_path='models')
    model.load("240424_entity_resolution_model.json")
    conn = sqlite3.connect('data/grant_npi.db')
    query = '''SELECT DISTINCT last_name FROM grants;'''
    names = pd.read_sql(query,con=conn)
    gids = np.ndarray([])
    nids = np.ndarray([])
    out = np.ndarray([])
    for name in names['last_name']:
        if name.find("'") == -1:
            query = f'''SELECT * FROM grants WHERE last_name = '{name}';'''
            grants = pd.read_sql(query,con=conn)
            query = f'''SELECT * FROM npi WHERE last_name = '{name}';'''
            npi = pd.read_sql(query,con=conn)
            df = model_features.features(grants,npi) # Why is this giving NaNs
            gids = np.append(gids,grants['application_id'])
            nids = np.append(nids,npi['npi'])
            out = np.append(out,model.predict(features=df))
    df['is_match'] = out
    df['npi_id'] = nids
    df['grant_id'] = gids
    df.to_sql(df.loc[df['is_match']],con=conn)

if __name__ == "__main__":
    map()