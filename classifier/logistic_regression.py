from sklearn.linear_model import LogisticRegression as RegressionClassifier

from classifier.classifier import Classifier

class LogisticRegression(Classifier):
    def __init__(self):
        super().__init__(RegressionClassifier(random_state=0), useWeights = True, useCalibration=True)
        self.name = "logistic regression"