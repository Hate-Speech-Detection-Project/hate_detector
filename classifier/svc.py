from sklearn.svm import NuSVC

from classifier.classifier import Classifier

class SVC(Classifier):
    def __init__(self):
        super().__init__(NuSVC(), useWeights = True)
        self.name = "svc"