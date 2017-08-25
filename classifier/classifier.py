import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

from numpy import unique
from numpy import random

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
            # unique, counts = np.unique(ground_truth, return_counts=True)
            # classCounts = dict(zip(unique, counts))
            # balanced_samples = self.balanced_sample_maker(features.tocsr(), ground_truth, max(classCounts.values()))
            # self.model.fit(balanced_samples[0], balanced_samples[1])
        else:
            self.model.fit(features, ground_truth)

        # necessary for multiprocessing
        return self

    def predict(self, features):
        assert self.model is not None, 'Executed predict() without calling fit()'
        if self.useCalibration:
            return self.model.predict_proba(features)[:, 1]
        else:
            return self.model.predict(features)

    def prediction_to_binary(self, prediction):
        result = prediction > 0.2
        return result

    def getWeights(self, labelArray):
        unique, counts = np.unique(labelArray, return_counts=True)
        classCounts = dict(zip(unique, counts))

        weights = [None] * len(labelArray)

        for key, label in enumerate(labelArray):
            weights[key] = len(labelArray)/classCounts[label]

        return np.array(weights)

    def balanced_sample_maker(self, X, y, sample_size, random_seed=None):
        """ return a balanced data set by sampling all classes with sample_size 
            current version is developed on assumption that the positive
            class is the minority.

        Parameters:
        ===========
        X: {numpy.ndarrray}
        y: {numpy.ndarray}
        """
        uniq_levels = np.unique(y)
        uniq_counts = {level: sum(y == level) for level in uniq_levels}

        if not random_seed is None:
            np.random.seed(random_seed)

        # find observation index of each class levels
        groupby_levels = {}
        for ii, level in enumerate(uniq_levels):
            obs_idx = [idx for idx, val in enumerate(y) if val == level]
            groupby_levels[level] = obs_idx
        # oversampling on observations of each label
        balanced_copy_idx = []
        for gb_level, gb_idx in groupby_levels.items():
            over_sample_idx = np.random.choice(gb_idx, size=sample_size, replace=True).tolist()
            balanced_copy_idx+=over_sample_idx
        np.random.shuffle(balanced_copy_idx)

        return (X[balanced_copy_idx, :], y[balanced_copy_idx], balanced_copy_idx)