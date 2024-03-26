'''
A reusable class that saves a classifier with associated metadata
'''
import os
import json
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn.model_selection


class Classifier():
    def __init__(self, model_dir: str, model_type: str = 'xgb'):
        self.model = (self._initialize_xgb_model() if model_type == 'xgb' 
                      else self._initialize_random_forests())
        self.metadata = {}
        self.model_dir = model_dir

    def train(self, features: pd.DataFrame, labels: pd.Series):
        """Train our classifier with features to predict labels

        Args:
            features (pd.DataFrame): a dataframe of features--
                the components used for calculations which may be
                created from the data
            labels (pd.Series): a set of associated labels per row--
                the goal of the classifier
        """
        features, features_test, labels, labels_test = (
            sklearn.model_selection.train_test_split(
                features, labels, test_size=0.2))
        self.model.fit(features, labels.astype('float'))
        self.metadata['training_date'] = datetime.datetime.now().strftime('%Y%m%d')
        self.metadata['training_rows'] = len(labels)
        self.metadata['accuracy'] = self.assess(features_test, labels_test)

    def predict(self, 
                features: pd.DataFrame, 
                proba: bool = False) -> np.ndarray:
        """Use a trained model to predict the output

        Args:
            features (pd.DataFrame): the input features
            proba (bool, optional): whether to return probabilities. 
                Defaults to False.

        Returns:
            np.ndarray: true or false labels for every row in features
                or probabilities of true for every row in features
        """
        if len(self.metadata) == 0:
            raise ValueError('Model must be trained before predicting')
        
        if proba:
            return self.model.predict_proba(features)[:, 0]
        return self.model.predict(features)
    
    def assess(self, features: pd.DataFrame, labels: pd.Series) -> float:
        """Compute the accuracy of our model

        Args:
            features (pd.DataFrame): input features 
            labels (pd.Series): known labels

        Returns:
            float: the accuracy of our model
        """
        pred_labels = self.predict(features)
        return sklearn.metrics.mean_squared_error(labels, pred_labels)

    def save(self, filename: str, overwrite: bool = False):
        """Save filename location model_path on hard drive"""
        if len(self.metadata) == 0:
            raise ValueError('Model must be trained before saving')
        
        today = datetime.datetime.now().strftime('%y%m%d')
        if filename[:6] != today:
            filename = f'{today}_{filename}'
            
        if os.path.splitext(filename)[1] != '.json':
            filename = filename + '.json'
        
        # We are saving our file here.
        path = os.path.join(self.model_dir, filename)
        metadata_path = os.path.splitext(path)[0] + '_metadata.json'

        # Pickle is dangerous in rare cases because it depends on having
        # access to the class VERSION that was used for saving.

        if not overwrite and (os.path.exists(path) or 
                              os.path.exists(metadata_path)):
            raise FileExistsError('Cannot overwrite existing file')

        self.model.save_model(path)
        with open(metadata_path, 'w') as fo:
            json.dump(self.metadata, fo)

    def load(self, filename: str):
        """Load in a model filename with associated metadata from 
        model_dir"""
        path = os.path.join(self.model_dir, filename)
        metadata_path = os.path.splitext(path)[0] + '_metadata.json'
        self.model.load_model(path)

        with open(metadata_path) as fr:
            self.metadata = json.load(fr)

    def _initialize_xgb_model(self):
        """Create a new xgbclassifier"""
        return xgb.XGBRegressor()
    

if __name__=="__main__":
    c = Classifier()