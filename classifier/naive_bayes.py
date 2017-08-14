from sklearn.naive_bayes import MultinomialNB

from classifier.classifier import Classifier

class NaiveBayes(Classifier):
    def __init__(self):
        super().__init__(MultinomialNB(), useWeights = True)