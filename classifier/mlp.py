from sklearn.neural_network import MLPClassifier

from classifier.classifier import Classifier

class MLP(Classifier):
    def __init__(self):
        super().__init__(MLPClassifier(hidden_layer_sizes=(500, 100, 5), max_iter=20, early_stopping=True), useWeights = False)
        self.name = "mlp"
