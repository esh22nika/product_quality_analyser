import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from visionguard.storage.database import init_db

class InspectionRepository:
    """Provides CRUD operations for inspection records and inspector feedback."""

    def __init__(self, db_path: str | Path = "./artifacts/visionguard.db"):
        self.db_path = str(db_path)
        init_db(self.db_path)

    def insert_inspection(
        self,
        inspection_id: str,
        input_filename: str,
        input_type: str,
        category: str,
        model_name: str,
        image_score: float,
        decision_status: str,
        review_label: Optional[str] = None,
        artifacts_path: Optional[str] = None
    ) -> None:
        """Inserts a new inspection record."""
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO inspections (
                    inspection_id, timestamp, input_filename, input_type,
                    category, model_name, image_score, decision_status,
                    review_label, artifacts_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                inspection_id, now, input_filename, input_type,
                category, model_name, float(image_score), decision_status,
                review_label or "UNREVIEWED", artifacts_path or ""
            ))

    def update_feedback(self, inspection_id: str, label: str, notes: str = "") -> None:
        """Updates inspector feedback for an inspection."""
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE inspections
                SET review_label = ?
                WHERE inspection_id = ?
            """, (label, inspection_id))

            conn.execute("""
                INSERT INTO feedback (inspection_id, inspector_label, notes, timestamp)
                VALUES (?, ?, ?, ?)
            """, (inspection_id, label, notes, now))

    def list_inspections(
        self,
        category: Optional[str] = None,
        decision_status: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """Retrieves inspections with optional filtering."""
        query = "SELECT * FROM inspections WHERE 1=1"
        params: List[Any] = []

        if category and category != "All":
            query += " AND category = ?"
            params.append(category)

        if decision_status and decision_status != "All":
            query += " AND decision_status = ?"
            params.append(decision_status)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def clear_all(self) -> None:
        """Deletes all inspection records (retention policy)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM inspections")
            conn.execute("DELETE FROM feedback")
