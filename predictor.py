from sklearn.metrics import precision_recall_fscore_support
from scipy.sparse import hstack
from feature.text_features import TextFeatures
from feature.ngram_features import NGramFeatures
from feature.word2vec import Word2Vec
from classifier.svr import SVR
import numpy as np
import pandas as pd

class Predictor:
    def fit(self, df):
        '''
        Generate the features from the dataframe and fit the classifiers.
        '''
        feature_matrix = self.calculate_feature_matrix(df)
        print("...using", feature_matrix.shape[1], "features from", ", ".join([feature[0] for feature in self.features]))
        for classifier in self.classifier:
            classifier[1].fit(feature_matrix, self.ground_truth(df))

    def predict(self, df):
        '''
        Predict whether the given comments are inappropriate according the the
        previous training. Returns common quality metrics for all classifiers.
        '''
        
        # Generate the features and predict the results.
        feature_matrix = self.calculate_feature_matrix(df)
        predictions = pd.DataFrame()
        for classifier in self.classifier:
            predictions[classifier[0]] = classifier[1].predict(feature_matrix)

        # For each classifier, generate some metrics like recall.
        metrics = {}
        for classifier in self.classifier:
            scores = precision_recall_fscore_support(
                self.ground_truth(df),
                classifier[1].prediction_to_binary(predictions[classifier[0]]),
                average='binary'
            )
            metrics[classifier[0]] = dict(zip(['precision', 'recall', 'f-score', 'support'], scores))

        self._metrics = metrics

        return predictions

    def metrics(self):
        return self._metrics

    def ground_truth(self, df):
        return df['hate']

    def calculate_feature_matrix(self, df):
        feature_matrix = hstack([feature[1].extractFeatures(df) for feature in self.features])
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