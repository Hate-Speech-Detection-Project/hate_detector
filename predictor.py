import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from scipy.sparse import hstack
# from numpy import hstack
from feature.textfeatures.text_features import TextFeatures
from feature.user_features import UserFeatures
from feature.simple_text_features import SimpleTextFeatures
from feature.ngram_features import NGramFeatures
from feature.character_ngram_features import CharacterNGramFeatures
from scipy.sparse import csr_matrix
from feature.word2vec import Word2Vec
from classifier.svr import SVR
from classifier.svc import SVC
from classifier.naive_bayes import NaiveBayes
from sklearn.preprocessing import normalize, MaxAbsScaler
from classifier.ensemble import HybridEnsemble
from classifier.random_forest import RandomForest
from classifier.logistic_regression import LogisticRegression
from scheduler import Scheduler
from multiprocessing import Pool
import pickle
import code
import time


class Predictor:
    def fit(self, df):
        '''
        Generate the features from the dataframe and fit the classifiers.
        '''

        pool = Pool(processes=4)
        print(", ".join([feature[0] for feature in self.features]))
        feature_matrix = self.calculate_feature_matrix(df)
        print("...using", feature_matrix.shape, "features from", ", ".join([feature[0] for feature in self.features]))

        processes = [None] * len(self.classifier)
        for index, classifier in enumerate(self.classifier):
            processes[index] = classifier[1].fit(feature_matrix, self.ground_truth(df))
            # processes[index] = pool.apply_async(classifier[1].fit, (feature_matrix, self.ground_truth(df)))

        for index, trainedClassifier in enumerate(processes):
            classifier = trainedClassifier # .get()
            self.classifier[index] = (classifier.name, classifier,)
            # f = open('trained_classifiers/' + str(df.shape[0]) + '_' + classifier.name + '_' + str(int(time.time())) + '.pickle', 'wb')
            # pickle.dump(classifier, f, protocol=4)

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
            hate_threshold = np.percentile(predictions[classifier[0]], 20)
            scores = precision_recall_fscore_support(
                self.ground_truth(df),
                (predictions[classifier[0]] > hate_threshold).astype(int),
                average='binary'
            )
            metrics[classifier[0]] = dict(zip(['precision', 'recall', 'f-score', 'support'], scores))

        self._metrics = metrics

        with open("measurements.txt", "a") as measurements_file:
            measurements_file.write(", ".join([feature[0] for feature in self.features]) + "\n\n")
            for threshold in range(0, 100, 10):
                hate_threshold = np.percentile(predictions, threshold)
                print("Threshold", hate_threshold)
                scores = precision_recall_fscore_support(
                    df['hate'],
                    (predictions > hate_threshold).astype(int),
                    average='binary'
                )
                print("Scores:", dict(zip(['precision', 'recall', 'f-score', 'support'], scores)))
                measurements_file.write("Threshold" + str(hate_threshold) + "\n")
                measurements_file.write(str(dict(zip(['precision', 'recall', 'f-score', 'support'], scores))) + "\n")
            measurements_file.write("#" * 80 + "\n\n")

        return predictions

    def metrics(self):
        return self._metrics

    def ground_truth(self, df):
        return df['hate']

    def calculate_feature_matrix(self, df):
        # feature_matrix = hstack([feature[1].extractFeatures(df) for feature in self.features])
        feature_matrix = self.features[0][1].extractFeatures(df)
        scaler = MaxAbsScaler()
        scaled_feature_matrix = scaler.fit_transform(feature_matrix)
        normalize(scaled_feature_matrix, norm='l2', axis=0, copy=False)
        self.feature_matrix = feature_matrix
        # code.interact(local=locals())
        # feature_matrix.data = np.nan_to_num(feature_matrix.data)
        # feature_matrix.eliminate_zeros()
        return feature_matrix

    def __init__(self):
        self.scheduler = Scheduler()

        self.features = [
            # ('text_features', TextFeatures()), # DB instance needed for these features
            ('character_ngram_features', CharacterNGramFeatures()),
            ('simple_text_features', SimpleTextFeatures()),
            ('word2vec', Word2Vec()),
            # ('user_features', UserFeatures()),
            ('ngram_features', NGramFeatures()),
        ]

        self.classifier = [
            # ('svr', SVR()),
            # ('svc', SVC()),
            # ('random forest', RandomForest()),
            ('logistic regression', LogisticRegression()),
            # ('naive_bayes', NaiveBayes()) # currently not working
        ]
        # self.classifier += [("ensemble_median", HybridEnsemble(self.classifier, 'median'))]
