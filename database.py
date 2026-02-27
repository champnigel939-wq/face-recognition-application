import sqlite3
from sqlite3 import Error

class Database:
    def __init__(self, db_file):
        """ create a database connection to a SQLite database """
        self.connection = None
        try:
            self.connection = sqlite3.connect(db_file)
        except Error as e:
            print(e)

    def create_table(self):
        """ create a table for storing face encodings and recognition history """
        try:
            sql_create_encodings_table = '''
            CREATE TABLE IF NOT EXISTS encodings (
                id INTEGER PRIMARY KEY,
                user_name TEXT NOT NULL,
                encoding BLOB NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            '''
            sql_create_history_table = '''
            CREATE TABLE IF NOT EXISTS recognition_history (
                id INTEGER PRIMARY KEY,
                user_name TEXT NOT NULL,
                recognized_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            '''
            cursor = self.connection.cursor()
            cursor.execute(sql_create_encodings_table)
            cursor.execute(sql_create_history_table)
        except Error as e:
            print(e)

    def insert_encoding(self, user_name, encoding):
        """ Insert a new encoding into the encodings table """
        sql = ''' INSERT INTO encodings(user_name, encoding)
                  VALUES(?,?) '''
        cur = self.connection.cursor()
        cur.execute(sql, (user_name, encoding))
        self.connection.commit()
        return cur.lastrowid

    def insert_recognition(self, user_name):
        """ Insert a new recognition event into the recognition history table """
        sql = ''' INSERT INTO recognition_history(user_name)
                  VALUES(?) '''
        cur = self.connection.cursor()
        cur.execute(sql, (user_name,))
        self.connection.commit()
        return cur.lastrowid

    def close(self):
        """ Close the database connection """
        if self.connection:
            self.connection.close()

if __name__ == '__main__':
    db = Database('face_recognition.db')
    db.create_table()