"""SQLite database for progress tracking"""

import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class ProgressDatabase:
    """Manages SQLite database for learning progress"""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS concepts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        topic TEXT NOT NULL,
        difficulty TEXT NOT NULL,
        quiz_passed BOOLEAN DEFAULT 0,
        implementation_passed BOOLEAN DEFAULT 0,
        gpu_validated BOOLEAN DEFAULT 0,
        last_reviewed DATE,
        next_review DATE,
        review_interval INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS quiz_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        concept_id INTEGER NOT NULL,
        question_id TEXT NOT NULL,
        correct BOOLEAN NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        next_review DATE,
        review_interval INTEGER DEFAULT 1,
        FOREIGN KEY (concept_id) REFERENCES concepts(id)
    );

    CREATE INDEX IF NOT EXISTS idx_next_review ON quiz_results(next_review);

    CREATE TABLE IF NOT EXISTS gpu_jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT UNIQUE NOT NULL,
        concept TEXT NOT NULL,
        backend TEXT NOT NULL,
        submitted_at DATETIME NOT NULL,
        completed_at DATETIME,
        status TEXT NOT NULL,
        cost REAL DEFAULT 0.0,
        accuracy REAL,
        baseline_accuracy REAL,
        passed BOOLEAN DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS study_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at DATETIME NOT NULL,
        ended_at DATETIME,
        concepts_studied TEXT,
        activities TEXT
    );
    """

    def __init__(self, db_path: Path):
        """Initialize database connection and schema

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        self._init_schema()

    def _init_schema(self):
        """Create database schema if not exists"""
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def add_concept(self, name: str, topic: str, difficulty: str):
        """Add a new concept to track

        Args:
            name: Concept name (unique identifier)
            topic: Topic category
            difficulty: Difficulty level (easy, intermediate, hard)
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO concepts (name, topic, difficulty)
            VALUES (?, ?, ?)
        """, (name, topic, difficulty))
        self.conn.commit()

    def get_concept(self, name: str) -> Optional[Dict[str, Any]]:
        """Get concept by name

        Args:
            name: Concept name

        Returns:
            Concept data as dictionary or None
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM concepts WHERE name=?", (name,))
        row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    def update_concept_mastery(
        self,
        name: str,
        quiz_passed: Optional[bool] = None,
        implementation_passed: Optional[bool] = None,
        gpu_validated: Optional[bool] = None
    ):
        """Update concept mastery status

        Args:
            name: Concept name
            quiz_passed: Quiz completion status
            implementation_passed: Implementation completion status
            gpu_validated: GPU validation status
        """
        updates = []
        params = []

        if quiz_passed is not None:
            updates.append("quiz_passed = ?")
            params.append(int(quiz_passed))

        if implementation_passed is not None:
            updates.append("implementation_passed = ?")
            params.append(int(implementation_passed))

        if gpu_validated is not None:
            updates.append("gpu_validated = ?")
            params.append(int(gpu_validated))

        if not updates:
            return

        params.append(name)
        query = f"UPDATE concepts SET {', '.join(updates)} WHERE name=?"

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        self.conn.commit()

    def get_all_concepts(self) -> list:
        """Get all concepts

        Returns:
            List of concept dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM concepts ORDER BY topic, name")
        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        """Close database connection"""
        self.conn.close()
