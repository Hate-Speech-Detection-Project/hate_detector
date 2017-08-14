from brew.base import Ensemble
from brew.base import EnsembleClassifier
from brew.combination.combiner import Combiner

from classifier.classifier import Classifier

class HybridEnsemble(Classifier):
    def __init__(self, classifierList, combiningMethod):
        classifiers = [None] * (len(classifierList))
        for key, tuple in enumerate(classifierList):
          classifiers[key] = tuple[1]

        hybridEnsemble = Ensemble(classifiers = classifiers)
        hybridEnsembleClassifier = EnsembleClassifier(ensemble=hybridEnsemble, combiner=Combiner(combiningMethod))

        super().__init__(hybridEnsembleClassifier)
        self.name = "ensemble"