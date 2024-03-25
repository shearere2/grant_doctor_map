from grant_doctor_map.mapping import classifier, string_distance_features
from grant_doctor_map.grants import grants_reader
from grant_doctor_map.npi import npi_reader
import pandas as pd

sdf = string_distance_features.StringDistanceFeatures()
df = pd.read_csv('data/toy_training.csv')
print()
features = sdf.features_from_pairs(df)
c = classifier.Classifier('models')
print()
df = c.train(df.drop(columns=['is_match']),df['is_match'])
c.predict()
c.save('grants_classifier')