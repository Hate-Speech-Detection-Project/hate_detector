from sklearn.ensemble import RandomForestClassifier

from classifier.classifier import Classifier

class RandomForest(Classifier):
    def __init__(self):
        super().__init__(RandomForestClassifier(random_state=0))
        self.name = "random forest"