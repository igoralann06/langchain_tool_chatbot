import sqlite3

DB_PATH = 'database.db'

# Create a connection and set up the table
def init_bot():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bots (
                id INTEGER PRIMARY KEY,
                sid TEXT
            )
        ''')
        conn.commit()

def add_bot(sid):
    record = get_bot_by_id(sid)
    if(not record):
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO bots (sid) 
                VALUES (?)
            ''', (sid,))
            conn.commit()

            print("Question sid saved successfully!")

def get_bots():
    # Open a new connection and set row factory
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, sid FROM bots
        ''')
        rows = cursor.fetchall()

        # Return the result as a list of dictionaries
        return [{"id": row["id"], "sid": row["sid"]} for row in rows]
    
def delete_bot(sid):
    # Open a new connection and set row factory
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM bots WHERE sid = ?
        ''', (sid,))
        conn.commit()

# Function to get question by ID
def get_bot_by_id(sid):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, sid FROM bots WHERE sid = ?
        ''', (sid,))
        row = cursor.fetchone()

        if row:
            return {"id": row["id"], "sid": row["sid"]}
        else:
            return None  # Return None if no question is found for the given ID
