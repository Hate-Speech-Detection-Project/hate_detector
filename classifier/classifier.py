import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV

class Classifier:
    def __init__(self, algorithm, useCalibration = False, useWeights = False):
        self.algorithm = algorithm
        self.model = None
        self.name = "unnamed"
        self.useCalibration = useCalibration
        self.useWeights = useWeights
        self.classes_ = [0, 1]

    def fit(self, features, ground_truth):
        self.model = self.algorithm
        if self.useCalibration:
            self.model = CalibratedClassifierCV(self.algorithm)

        if self.useWeights:
            self.model.fit(features, ground_truth, sample_weight=self.getWeights(ground_truth))
        else:
            self.model.fit(features, ground_truth)

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