import sqlite3
from pathlib import Path
from typing import Optional

def init_db(db_path: str | Path = "./artifacts/visionguard.db") -> sqlite3.Connection:
    """Initializes SQLite database schema for local inspection history."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                input_filename TEXT NOT NULL,
                input_type TEXT NOT NULL,
                category TEXT NOT NULL,
                model_name TEXT NOT NULL,
                image_score REAL,
                decision_status TEXT NOT NULL,
                review_label TEXT,
                artifacts_path TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id TEXT NOT NULL,
                inspector_label TEXT NOT NULL,
                notes TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (inspection_id) REFERENCES inspections (inspection_id)
            )
        """)

    return conn
