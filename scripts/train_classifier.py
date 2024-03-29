from grant_doctor_map.mapping import model_features, string_distance_features
import pandas as pd

sdf = string_distance_features.StringDistanceFeatures()
df = pd.read_csv('data/toy_training.csv')
print()
features = sdf.features_from_pairs(df)
c = model_features.Classifier('models')
print()
df = c.train(df.drop(columns=['is_match']),['is_match'])
c.predict()
c.save('grants_classifier')