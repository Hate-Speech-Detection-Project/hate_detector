from feature.db_interface import DBInterface
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math

SIMILAR = 0
DIFFERENT = 90


class TopicFeatures:
    def get_cos_similarity_for_article(self, comment, article):
        cos_sim_in_degree = SIMILAR
        corpus = [comment.strip()]
        if article is str and len(article) > 0 and type(comment) is str:
            vector = TfidfVectorizer(min_df=1)
            corpus.extend([article])
            print(corpus)
            vector.fit(corpus)

            tfidf_comment = vector.transform([comment])
            tfidf_article = vector.transform([article])

            cos_sim = cosine_similarity(tfidf_comment, tfidf_article)
            cos_sim_in_degree = self._cos_to_degree(cos_sim)

        return cos_sim_in_degree


    def get_cos_similarity_for_hate_comments_of_article(self, comment, hate_comments):
        cos_sim_in_degree = DIFFERENT
        corpus = [comment.strip()]

        if len(hate_comments) != 0 and type(comment) is str:

            hate_comments_corpus = ''
            for hate_comment in hate_comments:
                hate_comments_corpus += ' ' + hate_comment

            corpus.extend([hate_comments_corpus])

            vector = TfidfVectorizer(min_df=1)
            vector.fit(corpus)

            tfidf_comment = vector.transform([comment])
            tfidf_hate_comments = vector.transform([hate_comments_corpus])

            cos_sim = cosine_similarity(tfidf_comment, tfidf_hate_comments)
            cos_sim_in_degree = self._cos_to_degree(cos_sim)

        return cos_sim_in_degree


    def _cos_to_degree(self, cos):
        if cos <= 1.0 and cos >= 0:
            return math.degrees(math.acos(cos))

        return 0
