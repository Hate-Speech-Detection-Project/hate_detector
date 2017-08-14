import pandas as pd
from predictor import Predictor
import nltk
import os
from analysis.roc import ROC

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
    train_df = pd.read_csv('data/' + dataset + '/train.csv', sep=',')
    test_df = pd.read_csv('data/' + dataset + '/test.csv', sep=',')
    return (train_df, test_df)

def execute(dataset='tiny'):
    print("Using dataset", dataset)
    print("Load Data...")
    train_df, test_df = load_data(dataset)
    predictor = Predictor()
    print("Fit...")
    predictor.fit(train_df)
    print("Predict...")
    result = predictor.predict(test_df)
    # print("Result:")
    # print(result)
    print("Metrics:")
    print(predictor.metrics())

    roc = ROC()
    for classifier in result.columns.values:
        roc.calculate(result[classifier], test_df['hate'])
        roc.print(dataset + ' using ' + classifier)

def main():
    init_nltk()
    datasets = [
        'small',
        #'1000',
        #'10000',
        #'stratified',
        #'stratified_1000',
        #'stratified_10000',
        #'stratified_30000',
        #'all'
    ]
    for dataset in datasets:
        execute(dataset)


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()