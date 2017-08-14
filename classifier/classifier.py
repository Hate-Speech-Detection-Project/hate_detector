import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV

class Classifier:
    def __init__(self, algorithm, useCalibration = False):
        self.algorithm = algorithm
        self.model = None
        self.useCalibration = useCalibration

    def fit(self, features, ground_truth):
        print("Features")
        print(np.min(features))

        if self.useCalibration == True:
            self.model = CalibratedClassifierCV(self.algorithm)
        else:
            self.model = self.algorithm

        self.model.fit(features, ground_truth, sample_weight=self.getWeights(ground_truth))

    def predict(self, features):
        assert self.model is not None, 'Executed predict() without calling fit()'
        return self.model.predict(features)

    def prediction_to_binary(self, prediction):
        result = prediction > 0.1
        return result

    def getWeights(self, labelArray):
        unique, counts = np.unique(labelArray, return_counts=True)
        classCounts = dict(zip(unique, counts))
        weights = [None] * len(labelArray)

        for key, label in enumerate(labelArray):
            weights[key] = len(labelArray)/classCounts[label]

        return np.array(weights)