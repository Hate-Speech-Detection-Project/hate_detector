from sklearn.svm import SVR as sklearn_svr

from classifier.classifier import Classifier

class SVR(Classifier):
    def __init__(self):
        super().__init__(sklearn_svr(kernel='rbf'), useWeights = True)
        self.name = "svr"