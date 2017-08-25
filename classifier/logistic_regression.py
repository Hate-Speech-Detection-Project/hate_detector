from sklearn.linear_model import LogisticRegression as LogisticRegressionClassifier

from classifier.classifier import Classifier

class LogisticRegression(Classifier):
    def __init__(self):
        super().__init__(LogisticRegressionClassifier(random_state=0), useWeights=True, useCalibration=True)
        self.name = "logistic regression"