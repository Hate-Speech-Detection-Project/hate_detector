from multiprocessing import freeze_support

import pandas as pd
from feature.textfeatures.text_features import TextFeatures
# import seaborn as sns
# import matplotlib.pyplot as plt
# import sys
# import numpy as np
from scipy.stats.stats import pearsonr


class TargetFeatureCorrelator:
    def __init__(self):

        # print('read in files')
        # trainDf = pd.read_csv('../data/full/train.csv', sep=',')
        # testDf = pd.read_csv('../data/full/test.csv', sep=',')
        # print('finished read')
        #
        # text_features = TextFeatureGenerator()
        # train_feature_matrix = text_features.calculate_features_with_dataframe(trainDf).T
        # test_feature_matrix = text_features.calculate_features_with_dataframe(testDf).T
        #
        # result_matrix = [[]]
        # for index, feature in enumerate(train_feature_matrix):
        #     naive_bayes = BagOfWordsClassifier()
        #     feature_vector = feature.reshape(len(feature), 1)
        #
        #     naive_bayes.fitFeatureMatrix(feature_vector, trainDf['hate'])
        #
        #     result = test_feature_matrix[index]
        #     result_vector = result.reshape(len(result), 1)
        #
        #     predicted = naive_bayes.testFeatureMatrix(result_vector)
        #
        #     # result_vector = np.asarray([int(elem) for elem in predicted]).reshape(len(predicted), 1)
        #     # result_matrix = np.append(feature_vector, result_vector, axis=1)
        #
        #     # print(result,[int(elem) for elem in predicted])
        #     print('correlation')
        #     print(pearsonr(result, [int(elem) for elem in predicted]))
        #
        #     # print(feature)
        #     # print(np.asarray([int(elem) for elem in predicted]))
        #     # print(np.corrcoef(np.asarray(feature), np.asarray([int(elem) for elem in predicted])))
        #     # self.show_correlation_heat_map(np.corrcoef(feature_vector, result_vector), trainDf)

        print('read in files')
        trainDf = pd.read_csv('../data/userfeatures/train.csv.augmented.csv', sep=',')
        testDf = pd.read_csv('../data/userfeatures/test.csv.augmented.csv', sep=',')
        print('finished read')

        user_features = UserFeatureGenerator()
        train_user_features = user_features.calculate_features_with_dataframe(trainDf).T
        test_user_features = user_features.calculate_features_with_dataframe(testDf).T

        train_matrix = user_features.calculate_features_with_dataframe(trainDf)[:,[0,1]]
        test_matrix = user_features.calculate_features_with_dataframe(testDf)[:,[0,1]]

        result_matrix = [[]]
        for index, feature in enumerate(train_user_features):
            naive_bayes = BagOfWordsClassifier()
            feature_vector = feature.reshape(len(feature), 1)
            print(train_matrix)
            naive_bayes.fitFeatureMatrix(train_matrix, trainDf['hate'])

            result = test_user_features[index]
            result_vector = result.reshape(len(result), 1)



            predicted = naive_bayes.testFeatureMatrix(test_matrix)

            print(*predicted)
            # result_vector = np.asarray([int(elem) for elem in predicted]).reshape(len(predicted), 1)
            # result_matrix = np.append(feature_vector, result_vector, axis=1)

            # print(result,[int(elem) for elem in predicted])
            print('correlation')
            print(pearsonr(result, [int(elem) for elem in predicted]))



    def show_correlation_heat_map(self, feature_matrix, df):
        # #EM = #ExclamationMarks
        # #QM = #QuestionMarks
        # #DW = #DistinctWords
        # #WIT = #WordsInTotal
        # LC = CommentLength

        # #IJ = #Interjections
        # #AV = #Adverbs
        # #PP = #PersonalPronouns
        # #DM = #Determiner (e.g. "jener", "solcher")
        # #AJ = #Adjectives

        # CoSS A = cos similarity with article
        # CoSS NAC = cos similarity with not appropriate comments for all hate-comments of the article
        # CoSS AC = cos similarity with appropriate comments for the article

        # frame = pd.DataFrame(data=feature_matrix, index=range(len(df.index)))
        #
        # frame.columns = ['# EM', '# QM']
        # correlation_matrix = frame.corr()

        ax = plt.axes()
        sns.heatmap(feature_matrix, ax=ax, cbar=False)

        ax.set_title('Text-Feature Correlations')
        plt.show()
        sys.exit()


def main():
    # correlator = TargetFeatureCorrelator()

    features = TextFeatures()

    trainDf = pd.read_csv('../../data/10000/train.csv', sep=',')
    testDf = pd.read_csv('../../data/tiny/test.csv', sep=',')
    features.extractFeatures(trainDf)

if __name__ == "__main__":
    freeze_support()
    main()
