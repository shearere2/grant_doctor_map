import pandas as pd
import numpy as np
import jarowinkler
# from sentence_transformers import SentenceTransformer
# txts = df[col].to_list()
# df[f'{col}_vec'] = model.embed(txts)

class FeatureExtractor():
    def __init__(self):
        # self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        pass

    def features(self, 
                 grantees: pd.DataFrame, 
                 providers: pd.DataFrame) -> pd.DataFrame:
        """Compute distance features from a pair of dataframes"""
        cols_to_lowercase = ['forename', 'last_name', 'city', 'state']
        for col in cols_to_lowercase:
            grantees[col] = grantees[col].str.lower()

        # If it's training data
        comb = pd.concat([grantees.add_suffix('_g'), providers.add_suffix('_p')], axis=1)

        # If it's testing data
        comb = grantees.add_suffix('_g').merge(providers.add_suffix('_p'), 
                              how='outer',
                              left_on='last_name_g',
                              right_on='last_name_p')
        
        comb['jw_dist_forename'] = comb.apply(lambda row: jw_dist(row['forename_g'],
                                                                  row['forename_p']), 
                                                                  axis=1)  # Force row-by-row
        comb['set_dist_forename'] = comb.apply(lambda row: set_dist(row['forename_g'],
                                                                  row['forename_p']), 
                                                                  axis=1)  # Force row-by-row
        
        comb['jw_dist_city'] = comb.apply(lambda row: jw_dist(row['city_g'],
                                                                  row['city_p']), 
                                                                  axis=1)  # Force row-by-row
        comb['set_dist_city'] = comb.apply(lambda row: set_dist(row['city_g'],
                                                                  row['city_p']), 
                                                                  axis=1)  # Force row-by-row
        
        comb['jw_dist_state'] = comb.apply(lambda row: jw_dist(row['state_g'],
                                                                  row['state_p']), 
                                                                  axis=1)  # Force row-by-row
        comb['set_dist_state'] = comb.apply(lambda row: set_dist(row['state_g'],
                                                                  row['state_p']), 
                                                                  axis=1)  # Force row-by-row






def jw_dist(v1: str, v2: str) -> float:
    if isinstance(v1, str) and isinstance(v2, str):
        return jarowinkler.jarowinkler_similarity(v1, v2)
    else:
        return np.nan


def set_dist(v1: str, v2: str) -> float:
    if isinstance(v1, str) and isinstance(v2, str):
        v1 = set(v1.split(' '))
        v2 = set(v2.split(' '))
        return len(v1.intersection(v2))/min(len(v1), len(v2))