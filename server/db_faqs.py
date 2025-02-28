import sqlite3

DB_PATH = 'database.db'

# Create a connection and set up the table
def init_faq():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faqs (
                id INTEGER PRIMARY KEY,
                question TEXT,
                answer TEXT
            )
        ''')
        conn.commit()

def add_faq(question, answer):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO faqs (question, answer) 
            VALUES (?, ?)
        ''', (question, answer))
        conn.commit()

        print("Faq saved successfully!")

def get_faqs():
    # Open a new connection and set row factory
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, question, answer FROM faqs
        ''')
        rows = cursor.fetchall()

        # Return the result as a list of dictionaries
        return [{"id": row["id"], "question": row["question"], "answer": row["answer"]} for row in rows]
