import sqlite3

DB_PATH = 'database.db'

# Create a connection and set up the table
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY,
                question TEXT,
                isAnswered BOOL
            )
        ''')
        conn.commit()

def add_question(question):
    # Open a new connection
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO questions (question, isAnswered) 
            VALUES (?, ?)
        ''', (question, False))
        conn.commit()

        print("Question saved successfully!")

def get_questions():
    # Open a new connection and set row factory
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, question, isAnswered FROM questions WHERE isAnswered = 0
        ''')
        rows = cursor.fetchall()

        # Return the result as a list of dictionaries
        return [{"id": row["id"], "question": row["question"], "isAnswered": row["isAnswered"]} for row in rows]

# Function to get question by ID
def get_question_by_id(question_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, question, isAnswered FROM questions WHERE id = ?
        ''', (question_id,))
        row = cursor.fetchone()

        if row:
            return {"id": row["id"], "question": row["question"], "isAnswered": row["isAnswered"]}
        else:
            return None  # Return None if no question is found for the given ID

# Function to update a question's 'isAnswered' status
def update_question_in_db(question_id, is_answered):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE questions 
            SET isAnswered = ? 
            WHERE id = ?
        ''', (is_answered, question_id))
        conn.commit()

        print(f"Question ID {question_id} updated successfully!")
