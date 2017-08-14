import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV

class Classifier:
    def __init__(self, algorithm, useCalibration = False):
        self.algorithm = algorithm
        self.model = None
        self.name = "unnamed"
        self.useCalibration = useCalibration
        self.classes_ = [0, 1]

    def fit(self, features, ground_truth):
        self.model = self.algorithm
        if self.useCalibration == True:
            self.model = CalibratedClassifierCV(self.algorithm)

        self.model.fit(features, ground_truth)

    def predict(self, features):
        assert self.model is not None, 'Executed predict() without calling fit()'
        return self.model.predict(features)

    def prediction_to_binary(self, prediction):
        result = prediction > 0.1
        return result