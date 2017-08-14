from sklearn import metrics
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import time

class ROC:

    def __init__(self):
        self.fpr = None
        self.tpr = None
        self.thresholds = None
        self.predicted = None
        self.ground_truth = None

    def calculate(self, predicted, ground_truth):
        self.predicted = predicted
        self.ground_truth = ground_truth
        self.fpr, self.tpr, self.thresholds = metrics.roc_curve(ground_truth, predicted)
        return (self.fpr, self.tpr, self.thresholds)

    def print(self, label):
        roc_auc = auc(self.fpr, self.tpr)

        plt.figure()
        lw = 2
        plt.plot(self.fpr, self.tpr, color='darkorange',
                 lw=lw, label='ROC {} curve (area = %0.2f)'.format(label, roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('figures/roc_' + str(int(time.time())) + str(label) + '.png')
        # plt.show()
