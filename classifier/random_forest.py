from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import pandas as pd

class RandomForest:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators = 100)

    def fit(self, features, ground_truth):
        self.model.fit(features, ground_truth.astype(int))

    def predict(self, features):
        assert self.model is not None, 'Executed predict() without calling fit()'
        prediction = self.model.predict(features)
        return prediction

    def prediction_to_binary(self, prediction):
        result = prediction > 0.1
        return result