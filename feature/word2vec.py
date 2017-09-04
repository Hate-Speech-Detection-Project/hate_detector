from gensim.models import Word2Vec as gensim_word2vec
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import numpy as np
import re

class Word2Vec:
    def __init__(self):
        # load model
        self.w2v_model = gensim_word2vec.load('model/word2vec/all_lowercased_stemmed')
        # initialize stemmer
        self.stemmer = SnowballStemmer('german')
        # grab stopword list
        self.stop = stopwords.words('german')

    def extractFeatures(self, df):
        data = self.remove_stop_and_stem(df['comment'])
        vectors = np.asarray(list(map(self.comment_to_vectors, data)))
        return vectors


    def text_to_wordlist(self, comment):
        try:
            comment_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', comment, flags=re.MULTILINE)
            comment_text = re.sub(r'<\/?em>', '', comment_text, flags=re.MULTILINE)
            comment_text = re.sub("[^a-zA-ZöÖüÜäÄß]"," ", comment_text)
            comment_text = re.sub("\s\s+"," ", comment_text)
            comment_text = comment_text.lower() + '. '
        except:
            comment_text = ''
        return comment_text

    def to_wordlist(self, data):
        return data.apply(self.text_to_wordlist)

    def remove_stopwords(self, data):
        return data.apply(lambda x: [item for item in str(x).split(' ') if item not in self.stop])

    def stem(self, data):
        return data.apply(lambda x: " ".join([self.stemmer.stem(y) for y in x]))

    def word_to_position(self, word):
        try:
            return self.w2v_model.wv[word]
        except:
            return -1

    def comment_to_vectors(self, comment):
        words = comment.split(' ')
        result = list(map(self.word_to_position, words))
        result = sum(result) / len(words)
        return result

    def remove_stop_and_stem(self, data):
        data = self.to_wordlist(data)
        data = self.remove_stopwords(data)
        data = self.stem(data)
        return data

