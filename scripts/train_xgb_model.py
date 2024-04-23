import numpy as np
import pandas as pd

from grant_doctor_map.mapping import entity_resolution_model,model_features


def create_xgb_model(features: pd.DataFrame, labels: np.ndarray, path: str):
    xgb_model = entity_resolution_model.EntityResolutionModel('Models')
    xgb_model.train(features, labels)
    xgb_model.save(path)

df = pd.read_csv('data/toy_training.csv')
labels = df['is_match']
grants = df[['grant_application_id','grant_forename','grant_last_name',
            'grant_city','grant_state','grant_country']]
grants = grants.rename(columns={'grant_application_id':'application_id',
                        'grant_forename':'forename',
                        'grant_last_name':'last_name',
                        'grant_city':'city','grant_state':'state',
                        'grant_country':'country'})
npi = df[['npi_npi','npi_forename',
         'npi_last_name','npi_city','npi_state','npi_country']]
npi = npi.rename(columns={'npi_npi':'npi','npi_forename':'forename',
         'npi_last_name':'last_name','npi_city':'city','npi_state':'state',
         'npi_country':'country'})
print()
df = model_features.features(grants,npi)
print()
features = df
create_xgb_model(features, labels, 'entity_resolution_model.json')
