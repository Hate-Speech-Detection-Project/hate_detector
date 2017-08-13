from sklearn.feature_extraction.text import CountVectorizer
from crawler.db_interface import DBInterface
from sklearn.decomposition import LatentDirichletAllocation
from gensim import corpora, models
import gensim
from scipy import stats


class TopicModeling:

    def __init__(self):
        self.stopwords = []
        for row in open("../data/stopwords_de.txt"):
            self.stopwords.append(row.replace('\n', '').replace('\r', ''))
        self.dbinterface = DBInterface()

    def train(self):
        articles = []
        for tuple in self.dbinterface.get_all_articles():
            articles.append(tuple[0])
        articles = self.remove_stopwords(articles)
        dictionary = corpora.Dictionary(articles)
        corpus = [dictionary.doc2bow(text) for text in articles]

        print('starting training')
        self.lda = gensim.models.ldamodel.LdaModel(corpus, num_topics= 200,  alpha='auto')
        self.lda.save('../data/model/lda.model')
        print('training finished')
        comment = "Ich vermute mal das Sie bei Pro Asyl oder einer ähnlichen Gruppe arbeiten. Zum Beispiel Identitätverschleierung unerheblich. Das kann man sicher sehr bezweifeln, zumal in letzten Jahr bis zu 80 % der Menschen ihre Dokumente verloren hatte und es gleichzeitig in Istanbul einen florierenden Markt für gefälschte Dokumente gab, konnte man selbst hier lesen. Wen das also unerheblich sein sollte, haben Sie sicher Fakten."
        article = 'Innenminister Thomas de MaiziÞre (CDU) hat seine Plõne prõzisiert, mit denen die Sicherheit in Deutschland erh÷ht werden soll. Im Zentrum steht dabei ein strikteres Aufenthaltsrecht. "F³r Auslõnder, die straffõllig geworden sind oder von denen eine Gefõhrdung der ÷ffentlichen Sicherheit ausgeht, will ich das Aufenthaltsrecht weiter verschõrfen", sagte de MaiziÞre.Dazu soll de MaiziÞres Vorstellung nach ein neuer Haftgrund eingef³hrt werden, der die Ausweisung von straffõllig gewordenen Auslõndern vereinfachen soll. Zudem m³sse darauf hingewirkt werden, dass das geltende Recht bei Abschiebungen auch tatsõchlich angewendet wird. "Wir brauchen schnellere Verfahren", sagte de MaiziÞre.á Wer seine Abschiebung selbst durch Straftaten oder Identitõtstõuschungen verhindere, solle nicht mehr geduldet, sondern in Zukunft als vollziehbar ausreisepflichtig behandelt werden. Dazu solle eine eigene "Bund-Lõnder-Taskforce" eingerichtet werden.Zugleich will de MaiziÞre stõrker gegen radikalisierte deutsche Staatsb³rger vorgehen. Wer eine doppelte Staatsb³rgerschaft besitze und im Ausland f³r terroristische Milizen kõmpfe, solle den deutschen Pass verlieren.Weitergehenden Forderungen der Unions-Innenminister der Lõnder erteilte de MaiziÞre eine Absage. Er wandte sich sowohl gegen eine Abschaffung der doppelten Staatsangeh÷rigkeit als auch gegen ein generelles Burka-Verbot. Dies sei "verfassungsrechtlich problematisch". "Man kann nicht alles verbieten, was man ablehnt", sagte der Minister. Die beiden Forderungen stehen in einem Entwurf zur sogenannten Berliner Erklõrung der Lõnder-Innenminister der Union.áá "Das war eine Ohrfeige f³r die Scharfmacher in der CDU/CSU", sagte SPD-Chef Sigmar Gabriel zu de MaiziÞres Verteidigung der doppelten Staatsb³rgerschaft. Es sei wichtig, dass sich der Innenminister klar gegen Aktionismus ausgesprochen habe.Gleichzeitig signalisierte Gabriel Gesprõchsbereitschaft ³ber die neuen Anti-Terror-Vorschlõge des Bundesinnenministers. "Die SPD ist bereit, ³ber alles zu reden, was dazu beitrõgt, die Sicherheit weiter zu erh÷hen", sagte Gabriel. "F³r populistische Schnellsch³sse stehen wir aber nicht zur Verf³gung."'
        bow_article = dictionary.doc2bow(article.lower().split(' '))
        bow_comment = dictionary.doc2bow(comment.lower().split(' '))


        get_topicdistribution_for_comment = self.lda.get_document_topics(bow_comment, minimum_probability=0)
        get_topicdistribution_for_commentarticle = self.lda.get_document_topics(bow_article, minimum_probability=0)

        print([tupel[0] for tupel in get_topicdistribution_for_comment])
        print([tupel[1] for tupel in get_topicdistribution_for_comment])
        print([tupel[1] for tupel in get_topicdistribution_for_commentarticle])
        print(stats.entropy([tupel[1] for tupel in get_topicdistribution_for_comment],
                            [tupel[1] for tupel in get_topicdistribution_for_commentarticle]))
        # self.get_diff_for_topics(self.lda.get_document_topics(bow, minimum_probability=0.00001),
        #                          self.lda.get_document_topics(bow_comment, minimum_probability=0.00001))


    def get_diff_for_topics(self, document_topics, comment_topics):
        probability_diffs = []
        for tupel in comment_topics:
            probability_for_topic_in_comment = tupel[1]
            probability_for_topic_in_document = self.get_topic_probability(document_topics,tupel[0])
            diff = abs(probability_for_topic_in_comment - probability_for_topic_in_document)
            probability_diffs.append(diff)

        average = sum(probability_diffs)/len(probability_diffs)

        print(average)


    def get_topic_probability(self, topic_distribution, topic_number):
        for tupel in topic_distribution:
            if topic_number == tupel[0]:
                return tupel[1]
        return 0



    def remove_stopwords(self, list):
        cleaned_list = []
        for item in list:
            item = self.remove_stopwords_for_text(item)
            cleaned_list.append(item)
        return cleaned_list

    def remove_stopwords_for_text(self, text):
        text_list = []
        for item in text.split(' '):
            item_lowered = item.lower()
            if item_lowered not in self.stopwords:
                text_list.append(item_lowered)
        return text_list


