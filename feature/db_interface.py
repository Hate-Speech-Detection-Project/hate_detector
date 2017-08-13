import psycopg2
import sys


class DBInterface:
    def __init__(self):
        try:
            self.conn = psycopg2.connect("dbname='hatespeech' user='postgres' host='localhost' password='admin'")
        except:
            print("Cannot connect to database.")
            sys.exit(0)


    def insert_article(self, article):
        cur = self.conn.cursor()
        query = """ INSERT INTO articles (id,heading,body,ressort,url) VALUES (%s, %s, %s, %s, %s) """
        data = (article.get_id(), article.get_heading(), article.get_body(), article.get_ressort(), article.get_url())
        cur.execute(query, data)
        self.commit_queries()

    def article_table_already_exists(self):
        cur = self.conn.cursor()

        cur.execute(
            """
                 SELECT EXISTS(SELECT 1 FROM information_schema.tables
                    WHERE table_catalog = 'hatespeech' AND table_schema = 'public'
                    AND table_name = 'articles');
            """
        )
        return bool(cur.fetchone()[0])

    def comments_table_already_exists(self):
        cur = self.conn.cursor()

        cur.execute(
            """
                 SELECT EXISTS(SELECT 1 FROM information_schema.tables
                    WHERE table_catalog = 'hatespeech' AND table_schema = 'public'
                    AND table_name = 'comments');
            """
        )
        return bool(cur.fetchone()[0])

    def get_urls(self):
        cur = self.conn.cursor()

        cur.execute(
            """
                 SELECT DISTINCT url FROM comments;
            """
        )
        return cur.fetchall()

    def create_articles_table(self):
        cur = self.conn.cursor()
        cur.execute(
            """
                  CREATE TABLE articles (
                        id INTEGER PRIMARY KEY,
                        heading TEXT,
                        body TEXT NOT NULL,
                        ressort TEXT NOT NULL,
                        url TEXT NOT NULL
                    )
            """
        )
        self.conn.commit()


    def get_all_ressorts(self):
        cur = self.conn.cursor()
        cur.execute(
            """
                  SELECT DISTINCT ressort FROM articles;
            """
        )
        self.conn.commit()
        return cur.fetchall()

    def get_all_articles(self):
        cur = self.conn.cursor()
        cur.execute(
            """
                  SELECT body FROM articles;
            """
        )
        self.conn.commit()
        return cur.fetchall()


    def get_all_articles_by_ressort(self, ressort):
        cur = self.conn.cursor()
        query = (
            """
                  SELECT body FROM articles WHERE ressort LIKE %s;
            """
        )
        data = [ressort]
        cur.execute(query, data)
        self.commit_queries()
        return cur.fetchall()

    def get_articlebody_by_url(self, url):
        cur = self.conn.cursor()
        query = (
            """
                  SELECT body  FROM articles WHERE url LIKE %s;
            """
        )
        data = [url]
        cur.execute(query, data)
        self.commit_queries()
        return cur.fetchone()


    def get_articlebody_with_limit(self, limit):
        cur = self.conn.cursor()
        query = (
            """
                  SELECT url  FROM articles LIMIT %s
            """
        )
        data = [limit]
        cur.execute(query, data)
        self.commit_queries()
        return cur.fetchall()

    def get_comments_with_limit(self, limit):
        cur = self.conn.cursor()
        query = (
            """
                  SELECT comment,url FROM comments WHERE hate = 't' LIMIT %s
            """
        )
        data = [limit]
        cur.execute(query, data)
        self.commit_queries()
        return cur.fetchall()

    def get_all_comments_for_article(self, url):
        cur = self.conn.cursor()
        query = (
            """
                  SELECT comment  FROM comments WHERE url LIKE %s;
            """
        )
        data = [url]
        cur.execute(query, data)
        self.commit_queries()
        return cur.fetchall()

    def get_comments_for_article_by_type(self, url, hate):
        cur = self.conn.cursor()
        query = (
            """
                  SELECT comment  FROM comments WHERE (url LIKE %s) AND hate = %s;
            """
        )
        data = [url, hate]
        cur.execute(query, data)
        self.commit_queries()
        return cur.fetchall()

    def delete_articles_table(self):
        cur = self.conn.cursor()
        cur.execute(
            """
                  DROP TABLE articles
            """
        )
        self.conn.commit()

    def commit_queries(self):
        self.conn.commit()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.conn:
            self.conn.close()
