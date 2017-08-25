import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras
import sys
import datetime
from timeit import default_timer as timer

"""
-- Please execute the following queries on your database before trying this script.
CREATE INDEX articles_ressort ON articles(ressort);
CREATE INDEX articles_url ON articles(url);
CREATE INDEX comments_cid ON comments(cid);
CREATE INDEX comments_created ON comments(created);
CREATE INDEX comments_url ON comments(url);
CREATE INDEX comments_uid ON comments(uid);
CREATE INDEX comments_hate_uid ON comments(uid) WHERE hate;
CREATE INDEX comments_hate_created ON comments(created) WHERE hate;
CREATE INDEX comments_hate_created_uid ON comments(created, uid) WHERE hate;
CREATE INDEX comments_uid_created ON comments(uid, created);
CREATE INDEX comments_url_created ON comments(url, created);
CREATE INDEX comments_url_created_uid ON comments(url, created, uid);
ALTER TABLE comments ADD COLUMN ressort TEXT;
-- Attention, takes pretty long (> 20 minutes, potentially)
UPDATE comments SET ressort = (SELECT MAX(ressort) FROM articles WHERE comments.url = articles.url);
CREATE INDEX comments_ressort ON comments(ressort);
CREATE INDEX comments_uid_created_ressort ON comments(ressort, uid, created);
CREATE INDEX comments_hate_uid_created_ressort ON comments(ressort, uid, created) WHERE hate;
"""

BATCH_SIZE = 100000

# Output timestamp on every print.
def uprint(text):
  print('{:%Y-%m-%d %H:%M:%S}  '.format(datetime.datetime.now()) + text)
  sys.stdout.flush()

# Connect to the database.
uprint("Connecting to database...")
try:
    conn = psycopg2.connect("dbname='postgres' user='postgres' host='localhost' password='admin'")
except:
    uprint("Cannot connect to database.")
    sys.exit(0)
uprint("Connected to database.")

# Define the procedure for getting the thread depth of an article.
with conn.cursor() as cursor:
  cursor.execute(
    """
    CREATE OR REPLACE FUNCTION thread_depth (p_cid INTEGER)
      RETURNS INTEGER AS $$
    DECLARE
      counter INTEGER := 0;
      cur_pid INTEGER := 0;
    BEGIN 
      LOOP EXIT WHEN FALSE;
        SELECT pid FROM comments WHERE cid = p_cid INTO cur_pid;
        IF (cur_pid = 0 OR cur_pid IS NULL) THEN
          EXIT;
        ELSE
          SELECT cur_pid INTO p_cid;
          SELECT counter + 1 INTO counter;
        END IF;
      END LOOP;
      RETURN counter;
    END;
    $$ LANGUAGE plpgsql;
    """
  )

# Define the feature queries.
queries = [(
  # cid for later joining.
  'cid',
  """
  SELECT cid
  FROM temp_comments
  ORDER BY cid
  """
),(
  # Ressort.
  'ressort',
  """
  SELECT ressort
  FROM temp_comments
  ORDER BY cid
  """
),(
  # Thread depth.
  'thread_depth',
  """
  SELECT thread_depth(cid)
  FROM temp_comments
  ORDER BY cid
  """
),(
  # Time since last comment (by anyone) on same article.
  'time_since_last_comment',
  """
  SELECT outside.created - (
    SELECT COALESCE(MAX(inside.created), outside.created)
    FROM comments AS inside
    WHERE outside.url = inside.url
    AND outside.created > inside.created
  )
  FROM temp_comments AS outside
  ORDER BY cid
  """
),(
  # Time since last hate comment (by anyone) on same article.
  'time_since_last_hate_comment',
  """
  SELECT outside.created - (
    SELECT COALESCE(MAX(inside.created), outside.created)
    FROM comments AS inside
    WHERE outside.url = inside.url
    AND outside.created > inside.created
    AND inside.hate
  )
  FROM temp_comments AS outside
  ORDER BY cid
  """
),(
  # Time since last comment (by same user) on same article.
  'time_since_last_comment_same_user',
  """
  SELECT outside.created - (
    SELECT COALESCE(MAX(inside.created), outside.created)
    FROM comments AS inside
    WHERE outside.url = inside.url
    AND outside.created > inside.created
    AND outside.uid = inside.uid
  )
  FROM temp_comments AS outside
  ORDER BY cid
  """
),(
  # Time since last hate comment (by same user) on same article.
  'time_since_last_hate_comment_same_user',
  """
  SELECT outside.created - (
    SELECT COALESCE(MAX(inside.created), outside.created)
    FROM comments AS inside
    WHERE outside.url = inside.url
    AND outside.created > inside.created
    AND outside.uid = inside.uid
    AND inside.hate
  )
  FROM temp_comments AS outside
  ORDER BY cid
  """
),(
  # Time since last comment (by same user) on any article.
  'time_since_last_comment_same_user_any_article',
  """
  SELECT outside.created - (
    SELECT COALESCE(MAX(inside.created), outside.created)
    FROM comments AS inside
    WHERE outside.created > inside.created
    AND outside.uid = inside.uid
  )
  FROM temp_comments AS outside
  ORDER BY cid
  """
),(
  # Time since last hate comment (by same user) on any article.
  'time_since_last_hate_comment_same_user_any_article',
  """
  SELECT outside.created - (
    SELECT COALESCE(MAX(inside.created), outside.created)
    FROM comments AS inside
    WHERE outside.created > inside.created
    AND outside.uid = inside.uid
    AND inside.hate
  )
  FROM temp_comments AS outside
  ORDER BY cid
  """
),(
  # Time since creation of the corresponding article.
  'time_since_article',
  """
  SELECT temp_comments.created - COALESCE(EXTRACT(EPOCH FROM articles.publish_date), temp_comments.created)
  FROM temp_comments LEFT JOIN articles USING (url)
  ORDER BY cid
  """
),(
  # Number of comments by user at the time of writing the comment.
  'number_of_comments_by_user',
  """
  SELECT (
    SELECT COUNT(1)
    FROM comments AS inside
    WHERE inside.uid = outside.uid
    AND inside.created < outside.created
  )
  FROM temp_comments AS outside
  ORDER BY cid
  """
),(
  # Number of comments by user in the same ressort at the time of writing the comment.
  'number_of_comments_by_user_in_ressort',
  """
  SELECT (
    SELECT COUNT(1)
    FROM comments AS inside
    WHERE inside.uid = outside.uid
    AND inside.created < outside.created
    AND inside.ressort = outside.ressort
  )
  FROM temp_comments AS outside
  ORDER BY cid
  """
),(
  # Number of hate comments by user at the time of writing the comment.
  'number_of_hate_comments_by_user',
  """
  SELECT (
    SELECT COUNT(1)
    FROM comments AS inside
    WHERE inside.uid = outside.uid
    AND inside.created < outside.created
    AND inside.hate
  )
  FROM temp_comments AS outside
  ORDER BY cid
  """
),(
  # Number of hate comments by user in the same ressort at the time of writing the comment.
  'number_of_hate_comments_by_user_in_ressort',
  """
  SELECT (
    SELECT COUNT(1)
    FROM comments AS inside
    WHERE inside.uid = outside.uid
    AND inside.created < outside.created
    AND inside.ressort = outside.ressort
    AND inside.hate
  )
  FROM temp_comments AS outside
  ORDER BY cid
  """
)]

def execute_query(query):
  with conn.cursor() as cursor:
    cursor.execute(query)
    return [x[0] for x in cursor.fetchall()]

def augment(df, file, first=True):
  # Create a temporary view in the database with the comments in the dataset.
  with conn.cursor() as cursor:
    cursor.execute(
      """
      DROP VIEW IF EXISTS temp_comments
      """
    )
    cursor.execute(
      """
      CREATE TEMPORARY VIEW temp_comments AS
      SELECT * FROM comments
      WHERE cid IN (%s)
      ORDER BY created
      """ % (','.join([str(row['cid']) for index, row in df.iterrows()]))
    )
    
  # Execute the queries and cache the results.
  results = []
  for query in queries:
    start = timer()
    results.append(execute_query(query[1]))
    end = timer()
    uprint("   " + query[0] + ": " + "%.2f" % (end - start))

  # Create a nice dataframe with all the data.
  data_result = pd.DataFrame(results)
  data_result = data_result.transpose()
  data_result.columns = [query[0] for query in queries]
  df = df.merge(data_result, how='left', on='cid')

  # Output file.
  file_name = file + '.augmented.csv'
  if first:
    df.to_csv(file_name, encoding='utf-8', index=False)
  else:
    df.to_csv(file_name, encoding='utf-8', index=False, header=False, mode='a')

# Read the file we want to augment.
uprint("Loading file...")
file = sys.argv[1]
data = pd.read_csv(file, sep=',')
uprint("File loaded...")

# Split the dataset up into chunks determined by BATCH_SIZE.
row_count = len(data.index)
for index in range(0, row_count, BATCH_SIZE):
  augment(data.iloc[index:index+BATCH_SIZE, :], file, index == 0)
  uprint("Augmented %i rows starting at %i." % (BATCH_SIZE, index))

uprint("Done!")
