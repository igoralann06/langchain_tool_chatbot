import sqlite3
from db_chats import get_question

DB_PATH = 'database.db'

# Create a connection and set up the table
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY,
                sid TEXT
            )
        ''')
        conn.commit()

def add_question(sid):
    record = get_question_by_id(sid)
    if(not record):
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO questions (sid) 
                VALUES (?)
            ''', (sid,))
            conn.commit()

            print("Question sid saved successfully!")

def get_questions():
    # Open a new connection and set row factory
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, sid FROM questions
        ''')
        rows = cursor.fetchall()

        # Return the result as a list of dictionaries
        return [{"id": row["id"], "sid": row["sid"], "question": get_question(row["sid"])[0]["text"]} for row in rows]
    
def delete_question(sid):
    # Open a new connection and set row factory
    print(sid)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM questions WHERE sid = ?
        ''', (sid,))
        conn.commit()

# Function to get question by ID
def get_question_by_id(sid):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, sid FROM questions WHERE sid = ?
        ''', (sid,))
        row = cursor.fetchone()

        if row:
            return {"id": row["id"], "sid": row["sid"]}
        else:
            return None  # Return None if no question is found for the given ID
