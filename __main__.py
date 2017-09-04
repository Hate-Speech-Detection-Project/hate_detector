import matplotlib
matplotlib.use('Agg')
import pandas as pd
from predictor import Predictor
import nltk
import os
from analysis.roc import ROC
import pprint
import numpy as np

def init_nltk():
    if not os.path.exists('nltk'):
        os.makedirs('nltk')
    nltk.data.path.append(os.getcwd() + '/nltk')
    dependencies = ['corpora/stopwords']
    for package in dependencies:
        try:
            nltk.data.find(package)
        except LookupError:
            nltk.download(package, os.getcwd() + '/nltk')

def load_data(dataset='tiny'):
    train_df = pd.read_csv('data/' + dataset + '/train.csv', sep=',').dropna(subset=['comment', 'url'])
    test_df = pd.read_csv('data/' + dataset + '/test.csv', sep=',').dropna(subset=['comment', 'url'])

    train_df['ressort'].fillna('unknown', inplace=True)
    test_df['ressort'].fillna('unknown', inplace=True)

    fast_learning = True
    # Reduce sample set to stratified for faster learning
    if fast_learning:
        np.random.seed(42)
        train_df = train_df.reindex(np.random.permutation(train_df.index))

        samples = train_df[train_df['hate'] == True].shape[0]
        print("Samples: ", samples*2)
        X = pd.concat([train_df[train_df['hate'] == True].head(samples), train_df[train_df['hate'] == False].head(samples)])
        X = X.reindex(np.random.permutation(X.index))
    else:
        X = train_df

    return (X, test_df)

def execute(dataset='tiny'):
    print("Using dataset", dataset)
    print("Load Data...")
    train_df, test_df = load_data(dataset)
    predictor = Predictor()
    print("Fit... (to speed up, you can comment out features and classifier in predictor.py)")
    predictor.fit(train_df)
    print("Predict...")
    result = predictor.predict(test_df)
    # print("Result:")
    # print(result)
    print("Metrics:")
    pprint.PrettyPrinter().pprint(predictor.metrics())

    roc = ROC()
    for classifier in result.columns.values:
        roc.calculate(result[classifier], test_df['hate'])
        roc.print(dataset + ' using ' + classifier)

    result.to_csv('results/' + dataset + '.csv')

def main():
    init_nltk()
    datasets = [
       'test'
    ]
    for dataset in datasets:
        execute(dataset)


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
