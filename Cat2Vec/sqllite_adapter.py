import sqlite3
import os

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Exception as e:
        print(e)

    return conn

def get_articles(conn, limit=0):
    cur = conn.cursor()
    if limit > 0:
        cur.execute("SELECT * FROM articles LIMIT " + str(limit))
    else:
        cur.execute("SELECT * FROM articles")

    return cur.fetchall()

def set_up_db(db):
    if os.path.exists(db):
        print('Database exists, assume schema does, too.')
    else:
        print('Need to create schema')
        print('Creating schema...')
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("create table articles (Id INTEGER PRIMARY KEY, filename, sentiment, category, text)")
        cur.execute("CREATE UNIQUE INDEX uni_article on articles (filename, category)")
        conn.close()

def save_to_db(db, filename, sentiment, category, text):

    conn = db
    with conn:
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO articles (Id,filename,sentiment,category,text)\
                VALUES(?,?,?,?,?)",
                        (None, filename,sentiment, category, text))
        except sqlite3.IntegrityError:
            print('Record already inserted with title %s ')
