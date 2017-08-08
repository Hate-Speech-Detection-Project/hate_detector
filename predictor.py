from feature.text_features import TextFeatures
from feature.ngram_features import NGramFeatures
from feature.word2vec import Word2Vec
from classifier.svr import SVR
import numpy as np
import pandas as pd

class Predictor:
    # Get all features from feature combiner and train the classifier(s)
    def fit(self, df):
        feature_matrix = self.calculate_feature_matrix(df)
        print("...using", feature_matrix.shape[1], "features from", ", ".join([feature[0] for feature in self.features]))
        for classifier in self.classifier:
            classifier[1].fit(feature_matrix, self.ground_truth(df))

    # Predict if comments in dataframe are hate
    # returns dataframe containing one column with the result of each classifier and a column of the combined result
    def predict(self, df):
        feature_matrix = self.calculate_feature_matrix(df)
        result = pd.DataFrame()
        for classifier in self.classifier:
            result[classifier[0]] = classifier[1].predict(feature_matrix)
        return result

    def ground_truth(self, df):
        return df['hate']

    def calculate_feature_matrix(self, df):
        feature_matrix = np.concatenate([feature[1].extractFeatures(df) for feature in self.features], 1)
        return feature_matrix

    def __init__(self):
        self.features = [
            ('text_features', TextFeatures()),
            ('word2vec', Word2Vec()),
            ('ngram_features', NGramFeatures())
        ]

        self.classifier = [
            ('svr', SVR())
        ]