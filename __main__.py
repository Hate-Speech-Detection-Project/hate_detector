import pandas as pd
from predictor import Predictor
import nltk
import os

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

def main():
    init_nltk()
    print("Load Data...")
    train_df, test_df = load_data()
    predictor = Predictor()
    print("Fit...")
    predictor.fit(train_df)
    print("Predict...")
    result = predictor.predict(test_df)
    print("Result:")
    print(result)

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()