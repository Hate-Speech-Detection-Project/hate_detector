from sklearn.ensemble import RandomForestClassifier

from classifier.classifier import Classifier

class RandomForest(Classifier):
    def __init__(self):
        super().__init__(RandomForestClassifier(n_estimators = 100), useWeights = True)
        self.name = "random forest"