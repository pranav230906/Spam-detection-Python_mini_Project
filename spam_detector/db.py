# Placeholder for db.py
# db.py
import sqlite3
import os
from typing import Optional, List, Tuple

class Database:
    def __init__(self, path="data/messages.db"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.path)

    def _init_db(self):
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                true_label TEXT,
                predicted_label TEXT NOT NULL,
                predicted_prob REAL,
                model TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """)
            conn.commit()

    def insert_message(self, text: str, true_label: Optional[str], predicted_label: str, predicted_prob: float, model: str):
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO messages (text, true_label, predicted_label, predicted_prob, model)
            VALUES (?, ?, ?, ?, ?)
            """, (text, true_label, predicted_label, float(predicted_prob), model))
            conn.commit()
            return cur.lastrowid

    def query_messages(self, filter_label: Optional[str]=None) -> List[Tuple]:
        with self._connect() as conn:
            cur = conn.cursor()
            if filter_label:
                cur.execute("SELECT id, text, true_label, predicted_label, predicted_prob, model, timestamp FROM messages WHERE predicted_label = ? ORDER BY timestamp DESC", (filter_label,))
            else:
                cur.execute("SELECT id, text, true_label, predicted_label, predicted_prob, model, timestamp FROM messages ORDER BY timestamp DESC")
            return cur.fetchall()
