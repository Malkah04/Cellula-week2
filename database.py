import sqlite3
from datetime import datetime


DATABASE_NAME = "predictions.db"

def get_connection():
    connection = sqlite3.connect(DATABASE_NAME)
    return connection

def create_table():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            input_type TEXT NOT NULL,
            input_text TEXT,
            model_type TEXT,
            image_path TEXT,
            image_caption TEXT,
            predicted_label TEXT,
            confidence REAL,
            created_at TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def save_predictions(
    input_type,
    input_text,
    model_type,
    image_path,
    image_caption,
    predicted_label,
    confidence
):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            input_type,
            input_text,
            model_type,
            image_path,
            image_caption,
            predicted_label,
            confidence,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        input_type,
        input_text,
        model_type,
        image_path,
        image_caption,
        predicted_label,
        confidence,
        datetime.now().isoformat()
    ))

    connection.commit()
    connection.close()


def get_predictions():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            input_type,
            input_text,
            model_type,
            image_path,
            image_caption,
            predicted_label,
            confidence,
            created_at

        FROM predictions

        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    connection.close()

    return rows