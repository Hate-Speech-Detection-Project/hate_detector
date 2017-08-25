from feature.db_interface import DBInterface
from gensim import corpora, models
from scipy import stats
import warnings


class TopicModeling:
    def initialiseModel(self):
        self.stopwords = []
        for row in open("data/stopwords_de.txt"):
            self.stopwords.append(row.replace('\n', '').replace('\r', ''))

        try:
            self.dict = corpora.Dictionary().load('model/ldamodel/dictionary.dict')
            self.lda = models.LdaModel.load('model/ldamodel/lda.model')
            return True
        except:  # handle other exceptions such as attribute errors
            warnings.warn(
                'Could not open dictionary or lda-model. Maybe it does not exist. Train and Save it with "trainAndSaveModel"' +
                ' TopicModel-Feature is not used.')
            return False

    def saveDict(self):
        articles = []
        for tuple in self.dbinterface.get_all_articles():
            articles.append(tuple[0])
        articles = self.remove_stopwords(articles)
        dictionary = corpora.Dictionary(articles)
        dictionary.save("model/ldamodel/dictionary.dict")

    def calculateKullbackLeibnerDivergence(self, comment, article_url):
        dbinterface = DBInterface()
        kullbackLeiblerDivergence = 0
        article = dbinterface.get_articlebody_by_url(article_url)
        if not article is None:
            article_body = article[0]
            bow_article = self.dict.doc2bow(article_body.lower().split(' '))
            bow_comment = self.dict.doc2bow(comment.lower().split(' '))
            get_topicdistribution_for_comment = self.lda.get_document_topics(bow_comment, minimum_probability=0)
            get_topicdistribution_for_commentarticle = self.lda.get_document_topics(bow_article, minimum_probability=0)
            kullbackLeiblerDivergence = stats.entropy([tupel[1] for tupel in get_topicdistribution_for_comment],
                                                      [tupel[1] for tupel in get_topicdistribution_for_commentarticle])

        return kullbackLeiblerDivergence

    def trainAndSaveModel(self):
        dbinterface = DBInterface()
        articles = []
        for tuple in dbinterface.get_all_articles():
            articles.append(tuple[0])
        articles = self.remove_stopwords(articles)
        dictionary = corpora.Dictionary(articles)

        # calculate bow and save the corpus
        dictionary.save("model/ldamodel/dictionary.dict")

        corpus = [dictionary.doc2bow(text) for text in articles]

        print('starting training')
        self.lda = models.ldamodel.LdaModel(corpus, num_topics=200, alpha='auto')
        # save the trained model
        self.lda.save('model/ldamodel/lda.model')
        print('training finished')

    def get_diff_for_topics(self, document_topics, comment_topics):
        probability_diffs = []
        for tupel in comment_topics:
            probability_for_topic_in_comment = tupel[1]
            probability_for_topic_in_document = self.get_topic_probability(document_topics, tupel[0])
            diff = abs(probability_for_topic_in_comment - probability_for_topic_in_document)
            probability_diffs.append(diff)

        average = sum(probability_diffs) / len(probability_diffs)

        print(average)

    def get_topic_probability(self, topic_distribution, topic_number):
        for tupel in topic_distribution:
            if topic_number == tupel[0]:
                return tupel[1]
        return 0

    def remove_stopwords(self, list):
        cleaned_list = ['notempty']  #fix error when it only contains stopwords
        for item in list:
            if item is not None:
                item = self.remove_stopwords_for_text(item)
            else:
                item = ''
            cleaned_list.append(item)
        return cleaned_list

    def remove_stopwords_for_text(self, text):
        text_list = []
        for item in text.split(' '):
            item_lowered = item.lower()
            if item_lowered not in self.stopwords:
                text_list.append(item_lowered)
        return text_list
