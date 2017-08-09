from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class NGramFeatures:

  def __init__(self):
    self.count_vectorizer = CountVectorizer(ngram_range=(1,3))
    self.tfidf_transformer = TfidfTransformer()
    self.first = True

  def extractFeatures(self, df):
    features = None
    if self.first:
      counts = self.count_vectorizer.fit_transform(df["comment"])
      features = self.tfidf_transformer.fit_transform(counts)
      self.first = False
    else:
      counts = self.count_vectorizer.transform(df["comment"])
      features = self.tfidf_transformer.transform(counts)

    return features