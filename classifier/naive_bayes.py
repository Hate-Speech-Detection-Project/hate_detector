from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd

class NaiveBayes:
    def __init__(self):
        self.nb = MultinomialNB()
        self.model = None

    def fit(self, features, ground_truth):
        print("Features")
        print(np.min(features))
        self.model = self.nb.fit(features, ground_truth)

    def predict(self, features):
        assert self.model is not None, 'Executed predict() without calling fit()'
        return self.model.predict(features)

    def prediction_to_binary(self, prediction):
        result = prediction > 0.1
        return result