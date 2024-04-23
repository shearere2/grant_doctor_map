import numpy as np
import pandas as pd

from grant_doctor_map.mapping import entity_resolution_model


def create_xgb_model(features: pd.DataFrame, labels: np.ndarray, path: str):
    xgb_model = entity_resolution_model.EntityResolutionModel('Models')
    xgb_model.train(features, labels)
    xgb_model.save(path)

df = pd.read_csv('data/toy_training.csv')
features, labels = df.drop('is_match',axis=1), df['is_match']
create_xgb_model(features, labels, 'entity_resolution_model.json')