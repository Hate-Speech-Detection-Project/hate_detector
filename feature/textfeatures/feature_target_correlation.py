from multiprocessing import freeze_support

import pandas as pd
from feature.textfeatures.text_features import TextFeatures

def main():
    # correlator = TargetFeatureCorrelator()

    features = TextFeatures()

    # trainDf = pd.read_csv('../../data/stratified_10000/train.csv', sep=',')
    trainDf = pd.read_csv('../../data/stratified_10000/train.csv', sep=',')
    testDf = pd.read_csv('../../data/tiny/test.csv', sep=',')
    features.extractFeatures(trainDf)
    # model = TopicModeling()
    # comment = "Ich vermute mal das Sie bei Pro Asyl oder einer ähnlichen Gruppe arbeiten. Zum Beispiel Identitätverschleierung unerheblich. Das kann man sicher sehr bezweifeln, zumal in letzten Jahr bis zu 80 % der Menschen ihre Dokumente verloren hatte und es gleichzeitig in Istanbul einen florierenden Markt für gefälschte Dokumente gab, konnte man selbst hier lesen. Wen das also unerheblich sein sollte, haben Sie sicher Fakten."
    # article = "http://www.zeit.de/2016/01/teilen-abgeben-konflikt"
    # model.calculateKullbackLeibnerDivergence(comment,article)
    # model.saveDict()


if __name__ == "__main__":
    freeze_support()
    main()
