import pandas as pd

from grant_doctor_map.mapping import db


def create_training_data_features_labels():
    '''Load known training data, extract values from db,
    convert to features.'''
    training = pd.read_csv('data/likely_grantee_provider_matches.csv')
    grantee_ids = ', '.join(training['g_id'])
    query = f'''SELECT last_name, forename
            FROM grants
            WHERE id IN ({grantee_ids})'''
    grantees = pd.read_sql()
    npi_ids = ', '.join(training['g_id'])
    query = f'''SELECT TRIM(LOWER(last_name)) AS last_name,
            forename
            FROM grants
            WHERE id IN ({grantee_ids})'''
    grantees = pd.read_sql()
