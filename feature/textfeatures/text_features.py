import time
from enum import Enum
from itertools import chain
from multiprocessing import Process, Manager, cpu_count
from threading import Thread
import numpy as np
from feature.textfeatures.topic_features import TopicFeatures
from textblob_de import TextBlobDE as TextBlob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from feature.textfeatures.topic_modeling import TopicModeling

CORE_THREADS_NUM = cpu_count()
RESULT_COUNT = 17


class Resultindices(Enum):
    LENGTH_OF_COMMENT, NUM_OF_WORDS, NUM_OF_DISTINCT_WORDS, \
    NUM_OF_QUESTIONMARKS, NUM_OF_EXCLAMATIONMARKS, NUM_OF_DOTS, NUM_OF_QUOTES, \
    NUM_OF_REPEATED_DOT, RATIO_CAPITALIZED, NUM_OF_HTTP, \
    NUM_OF_ADJECTIVES, NUM_OF_DETERMINER, NUM_OF_PERSONAL_PRONOUNS, NUM_OF_ADVERBS, \
    COS_SIMILARITY_ARTICLE, COS_SIMILARITY_HATE_COMMENTS, KULLBACK_LEIBLER_DIVERGENCE_TO_ARTICLE = range(
        RESULT_COUNT)


class TextFeatures:
    def __init__(self):
        self.topic_features = TopicFeatures()
        self.results = []

    def extractFeatures(self, old_df):
        print('Start extraction of text-features')
        self.results = []
        start_time = time.time()

        df = old_df[['comment', 'url']]
        self.df_parts = np.array_split(df, CORE_THREADS_NUM)


        tagged_comments = self._tagComments()
        self._calculateSemanticFeatures(tagged_comments)

        self._calculateCosSimilarity()

        self._calculateSyntaxFeatures(df)

        self._calculateFeatureFromTopicModel(old_df)


        print("--- Took %s seconds ---" % (time.time() - start_time))
        features = np.vstack(self.results).T
        self.show_correlation_heat_map(features, old_df)

        return features


    def _calculateSemanticFeatures(self, tagged_comments):
        threads = []

        threads.append(
            Thread(target=(
                lambda x, results, index: results.insert(index, list(map(
                    lambda x: TextFeatures._getCountOfWordsByTaggedList(x, ['JJ', 'JJS', 'JJR']), tagged_comments)))),
                args=(tagged_comments, self.results, Resultindices.NUM_OF_ADJECTIVES.value)))

        threads.append(
            Thread(target=(
                lambda x, results, index: results.insert(index, list(map(
                    lambda x: TextFeatures._getCountOfWordsByTaggedList(x, ['DT']), tagged_comments)))),

                args=(tagged_comments, self.results, Resultindices.NUM_OF_DETERMINER.value)))

        threads.append(
            Thread(target=(
                lambda x, results, index: results.insert(index, list(map(
                    lambda x: TextFeatures._getCountOfWordsByTaggedList(x, ['PRP']), tagged_comments)))),
                args=(tagged_comments, self.results, Resultindices.NUM_OF_PERSONAL_PRONOUNS.value)))

        threads.append(
            Thread(target=(
                lambda x, results, index: results.insert(index, list(map(
                    lambda x: TextFeatures._getCountOfWordsByTaggedList(x, ['RB', 'RBS']), tagged_comments)))),
                args=(tagged_comments, self.results, Resultindices.NUM_OF_ADVERBS.value)))

        self.startProcessesAndJoinThem(threads)
        threads.clear()

    def _calculateCosSimilarity(self):

        # calculate cos-similarity between article and comment
        manager = Manager()
        return_dict = manager.dict()

        processes = []

        for index, df_part in enumerate(self.df_parts):
            p = Process(target=self._calculate_article_cos_similarity,
                        args=(df_part, return_dict, index))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        processes.clear()
        result = list(chain.from_iterable(return_dict.values()))
        self.results.insert(Resultindices.COS_SIMILARITY_ARTICLE.value, result)

        # _______________________________________________________________________
        # calculate cos-similarity between hate-comments of article and comment

        manager = Manager()
        return_dict = manager.dict()

        processes = []

        for index, df_part in enumerate(self.df_parts):
            p = Process(target=self._calculate_hate_cos_similarity,
                        args=(df_part, return_dict, index))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        processes.clear()
        result = list(chain.from_iterable(return_dict.values()))
        self.results.insert(Resultindices.COS_SIMILARITY_HATE_COMMENTS.value, result)

    def _calculateSyntaxFeatures(self, df):

        threads = []

        threads.append(
            Thread(target=(
                lambda df, results, index: results.insert(index, df['comment'].apply(lambda x: int(len(x))))),
                args=(df, self.results, Resultindices.LENGTH_OF_COMMENT.value)))

        threads.append(
            Thread(target=(
                lambda df, results, index: results.insert(index, df['comment'].apply(lambda x: int(len(x.split()))))),
                args=(df, self.results, Resultindices.NUM_OF_WORDS.value)))

        threads.append(
            Thread(target=(
                lambda df, results, index: results.insert(index,
                                                          df['comment'].apply(lambda x: int(len(set(x.split())))))),
                args=(df, self.results, Resultindices.NUM_OF_DISTINCT_WORDS.value)))

        threads.append(
            Thread(target=(
                lambda df, results, index: results.insert(index, df['comment'].apply(lambda x: int(x.count('?'))))),
                args=(df, self.results, Resultindices.NUM_OF_QUESTIONMARKS.value)))

        self.startProcessesAndJoinThem(threads)
        threads.clear()

        threads.append(
            Thread(target=(
                lambda df, results, index: results.insert(index, df['comment'].apply(lambda x: int(x.count('!'))))),
                args=(df, self.results, Resultindices.NUM_OF_EXCLAMATIONMARKS.value)))

        threads.append(
            Thread(target=(
                lambda df, results, index: results.insert(index, df['comment'].apply(lambda x: int(x.count('..'))))),
                args=(df, self.results, Resultindices.NUM_OF_REPEATED_DOT.value)))

        threads.append(
            Thread(target=(
                lambda df, results, index: results.insert(index, df['comment'].apply(lambda x: int(x.count('.'))))),
                args=(df, self.results, Resultindices.NUM_OF_DOTS.value)))

        threads.append(
            Thread(target=(
                lambda df, results, index: results.insert(index, df['comment'].apply(lambda x: int(x.count('"'))))),
                args=(df, self.results, Resultindices.NUM_OF_QUOTES.value)))

        threads.append(
            Thread(target=(
                lambda df, results, index: results.insert(index, df['comment'].apply(
                    lambda x: sum(1 for c in x if c.isupper()) / len(x)))),
                args=(df, self.results, Resultindices.RATIO_CAPITALIZED.value)))

        threads.append(
            Thread(target=(
                lambda df, results, index: results.insert(index, df['comment'].apply(lambda x: int(x.count('http'))))),
                args=(df, self.results, Resultindices.NUM_OF_HTTP.value)))

        self.startProcessesAndJoinThem(threads)
        threads.clear()

    def _applyPOSTaggin(self, df_part, result_dict, index):
        # text_blob_comments = df_part.apply(lambda x: TextBlob(x).tags)
        text_blob_comments = []
        for row in df_part['comment']:
            text_blob_comments.append(TextBlob(row).tags)
        result_dict[index] = text_blob_comments

    def startProcessesAndJoinThem(self, processes):
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    def _tagComments(self):
        manager = Manager()
        return_dict = manager.dict()
        processes = []

        for i, df_part in enumerate(self.df_parts):
            p = Process(target=self._applyPOSTaggin, args=(df_part, return_dict, i))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        tagged_comments = list(chain.from_iterable(return_dict.values()))
        return tagged_comments

    def _calculate_article_cos_similarity(self, df, cos_list, result_index):
        topic_features = TopicFeatures()
        list = []
        for index, row in df.iterrows():
            list.append(int(topic_features.get_cos_similarity_for_article(row['comment'], row['url'])))
        cos_list[result_index] = list

    def _calculate_hate_cos_similarity(self, df, cos_list, result_index):
        topic_features = TopicFeatures()
        list = []
        for index, row in df.iterrows():
            list.append(int(topic_features.get_cos_similarity_for_hate_comments_of_article(row['comment'], row['url'])))
        cos_list[result_index] = list

    def _calculateFeatureFromTopicModel(self, df):

        topic_model = TopicModeling()
        successfully_instantiated_model = topic_model.initialiseModel()

        if not successfully_instantiated_model:
            return

        list = []
        for index, row in df.iterrows():
            list.append(topic_model.calculateKullbackLeibnerDivergence(row['comment'], row['url']))

        self.results.insert(Resultindices.KULLBACK_LEIBLER_DIVERGENCE_TO_ARTICLE.value, list)

    def _applyTopicModel(self,df,result_list, index, topic_model):
        list = []
        for index, row in df.iterrows():
            list.append(topic_model.calculateKullbackLeibnerDivergence(row['comment'], row['url']))
        result_list[index] = list


    def show_correlation_heat_map(self, feature_matrix, df):
        # #EM = #ExclamationMarks
        # #QM = #QuestionMarks
        # #DW = #DistinctWords
        # # DW = # DistinctWords
        # LC = CommentLength
        # NM = NumOfWords
        # # D = NumOfDots
        # # Q = NumOfQuotes
        # # RD = NumOfRepeatedDots
        # RC = RatioCapitalized
        # # HTTP = NumOfHttp

        # #IJ = #Interjections
        # #AV = #Adverbs
        # #PP = #PersonalPronouns
        # #DM = #Determiner (e.g. "jener", "solcher")
        # #AJ = #Adjectives

        # CoSS A = cos similarity with article
        # CoSS NAC = cos similarity with non-appropriate comments for the article
        # KLDA = KullbackLeiblerDivergenceToArticle

        # TV = TargetVariable

        target_variable = df['hate'].apply(lambda x: int(x))
        feature_matrix = np.hstack((feature_matrix, np.reshape(target_variable, (len(df), 1))))

        frame = pd.DataFrame(data=feature_matrix, index=range(len(feature_matrix)))
        frame.columns = ['LC', 'NM', '# DW', '# QM', '# EM', '# D', '# Q', '# RD', 'RC', '# HTTP',
                         '# AJ', '# DM', '# PP', '# AV',
                         'COS_A', 'COS_NAC', 'KLDA', 'TV']
        correlation_matrix = frame.corr()
        np.savetxt("./textfeaturematrix.csv",correlation_matrix, delimiter=";")

        ax = plt.axes()
        sns.heatmap(correlation_matrix, ax=ax)


        ax.set_title('Text-Feature Correlations')
        plt.show()
        sys.exit()

    @staticmethod
    def _getCountOfWordsByTaggedList(tagged_list, tag_id_list):
        count = 0
        for tag in tagged_list:
            if tag[1] in tag_id_list:
                count = count + 1

        return count
