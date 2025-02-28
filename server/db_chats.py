import sqlite3

DB_PATH = 'database.db'

# Create a connection and set up the table
def init_chats():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY,
                text TEXT,
                sid TEXT,
                status BOOLEAN
            )
        ''')
        conn.commit()

def add_chat(text, sid, status):
    # Open a new connection
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO chats (text, sid, status) 
            VALUES (?, ?, ?)
        ''', (text, sid, status))
        conn.commit()

        print("Question saved successfully!")

def get_chats(sid):
    # Open a new connection and set row factory
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, text, sid, status FROM chats WHERE sid = ?
        ''', (sid,))
        rows = cursor.fetchall()

        # Return the result as a list of dictionaries
        return [{"id": row["id"], "text": row["text"], "sid": row["sid"], "status": row["status"]} for row in rows]
    
def get_question(sid):
    # Open a new connection and set row factory
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, text, sid, status FROM chats WHERE sid = ? AND status = 0 ORDER BY id DESC LIMIT 1
        ''', (sid,))
        rows = cursor.fetchall()

        # Return the result as a list of dictionaries
        return [{"id": row["id"], "text": row["text"], "sid": row["sid"], "status": row["status"]} for row in rows]
