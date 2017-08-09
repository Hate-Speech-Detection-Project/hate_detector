import numpy as np

class UserFeatures:

  def extractFeatures(self, df):
    features = np.vstack((
      df['time_since_last_comment'],
      df['time_since_last_comment_same_user'],
      df['time_since_last_hate_comment_same_user'],
      df['time_since_last_comment_same_user_any_article'],
      df['time_since_last_hate_comment_same_user_any_article'],
      df['number_of_comments_by_user'],
      df['number_of_hate_comments_by_user'],
      df['share_of_hate_comments_by_user']
    )).T

    return features
