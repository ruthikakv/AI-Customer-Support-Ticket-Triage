import sqlite3
import pandas as pd
from pathlib import Path

# Database location
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATABASE_PATH = PROJECT_DIR / "data" / "ticket_history.db"


# Create database and table
def initialize_database():

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ticket_history (

            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket TEXT NOT NULL,
            category TEXT NOT NULL,
            urgency TEXT NOT NULL,
            support_team TEXT NOT NULL,
            category_confidence REAL,
            urgency_confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
    """)

    connection.commit()

    connection.close()


# Save analyzed ticket
def save_ticket(ticket, category, urgency, support_team,
                category_confidence, urgency_confidence):

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO ticket_history
        (ticket, category, urgency, support_team,
         category_confidence, urgency_confidence)

        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        ticket,
        category,
        urgency,
        support_team,
        category_confidence,
        urgency_confidence
    ))

    connection.commit()

    connection.close()
    # Retrieve all saved tickets
def get_all_tickets():

    connection = sqlite3.connect(DATABASE_PATH)

    df = pd.read_sql_query(
        "SELECT * FROM ticket_history ORDER BY id DESC",
        connection
    )

    connection.close()

    return df