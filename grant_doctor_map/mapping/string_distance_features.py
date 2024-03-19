import fasttext
import jarowinkler 
import numpy as np
import pandas as pd


class StringDistanceFeatures():
    def __init__(self, ft_model_path: str = 'data/cc.en.50.bin'):
        self.ft_model = fasttext.load_model(ft_model_path)

    def combine_prediction_data(self, grants: pd.DataFrame, npi: pd.DataFrame) -> pd.DataFrame:
        """Combine grants and npi dataframes into pairs"""
        grants = grants.iloc[0:100].add_prefix('grant_')
        npi = npi.iloc[0:100].add_prefix('npi_')
        grants['merge_val'] = 1
        npi['merge_val'] = 1

        return grants.merge(npi, on='merge_val')

    def features_from_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes distance features from a dataframe of pairs
        of grant_ and npi_ data"""
        data_cols = df.columns

        df['jw_dist_last'] = df.apply(lambda row: 
                                 jarowinkler.jaro_similarity(row['grant_last_name'],
                                                            row['npi_last_name']), axis = 1)
        df['jw_dist_first'] = df.apply(lambda row: 
                                 jarowinkler.jaro_similarity(row['grant_forename'],
                                                            row['npi_forename']), axis = 1)
        df['match_city'] = df.apply(lambda row: 
                                     (row['grant_city'] == row['npi_city']), axis = 1)
        df['match_state'] = df.apply(lambda row: 
                                    (row['grant_state'] == row['npi_state']), axis = 1)
        
        for dataset in ['grant', 'npi']:
            for col in ['last_name', 'forename']:
                df[f'vec_{dataset}_{col}'] = df[f'{dataset}_{col}'].apply(
                    lambda x: self.ft_model.get_sentence_vector(x) 
                            if isinstance(x, str) else np.nan)

        df['ft_dist_last_name'] = df.apply(lambda row: np.linalg.norm((row['vec_grant_last_name']) - row['vec_npi_last_name']),
                                            axis = 1)
        
        return df.drop(columns=data_cols).drop(columns=[
            v for v in df.columns if 'vec' in v])


if __name__ == '__main__':
    from grants import grants_reader
    from npi import npi_reader
    grants_df = grants_reader.read_grants_year(2022)
    npi_df = npi_reader.read('data/npidata_pfile_20240205-20240211.csv')

    sdf = StringDistanceFeatures()
    comb_df = sdf.combine_prediction_data(grants_df, npi_df)
    features = sdf.features_from_pairs(comb_df)
    print(features)
