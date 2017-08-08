import numpy as np

class TextFeatures:

  def extractFeatures(self, df):
    total_length = df['comment'].apply(lambda x: len(x))
    num_of_words = df['comment'].apply(lambda x: len(x.split()))
    avg_length = df['comment'].apply(lambda x: np.average([len(a) for a in x.split()]))
    num_questions = df['comment'].apply(lambda x: x.count('?'))
    num_quote = df['comment'].apply(lambda x: x.count('"'))
    num_dot = df['comment'].apply(lambda x: x.count('.'))
    num_repeated_dot = df['comment'].apply(lambda x: x.count('..'))
    num_exclamation = df['comment'].apply(lambda x: x.count('!'))
    num_http = df['comment'].apply(lambda x: x.count('http'))
    num_negations = df['comment'].apply(lambda x: x.count('nicht') + x.count('nie') + x.count('weder') + x.count('nichts'))
    ratio_capitalized = df['comment'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x))

    features = np.vstack((
      total_length,
      num_of_words,
      avg_length,
      num_questions,
      num_quote,
      num_dot,
      num_repeated_dot,
      num_exclamation,
      num_http,
      num_negations,
      ratio_capitalized
    )).T

    return features
