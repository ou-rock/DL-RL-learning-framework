"""Database for progress tracking - supports SQLite (local) and PostgreSQL (production)"""

import os
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, Protocol, runtime_checkable
from datetime import datetime

# Try to import PostgreSQL support
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    psycopg2 = None


@runtime_checkable
class DatabaseBackend(Protocol):
    """Protocol defining database backend interface"""

    def add_concept(self, name: str, topic: str, difficulty: str) -> None: ...
    def get_concept(self, name: str) -> Optional[Dict[str, Any]]: ...
    def get_quiz_stats(self, concept_name: str) -> Dict[str, Any]: ...
    def update_concept_mastery(
        self, name: str,
        quiz_passed: Optional[bool] = None,
        implementation_passed: Optional[bool] = None,
        gpu_validated: Optional[bool] = None
    ) -> None: ...
    def get_all_concepts(self) -> list: ...
    def close(self) -> None: ...


class SQLiteBackend:
    """SQLite backend for local progress tracking"""

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
        """Initialize SQLite connection

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Create database schema if not exists"""
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def add_concept(self, name: str, topic: str, difficulty: str):
        """Add a new concept to track"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO concepts (name, topic, difficulty)
            VALUES (?, ?, ?)
        """, (name, topic, difficulty))
        self.conn.commit()

    def get_concept(self, name: str) -> Optional[Dict[str, Any]]:
        """Get concept by name"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM concepts WHERE name=?", (name,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_quiz_stats(self, concept_name: str) -> Dict[str, Any]:
        """Get aggregated quiz statistics for a concept"""
        cursor = self.conn.cursor()
        
        # Get concept ID first
        cursor.execute("SELECT id FROM concepts WHERE name=?", (concept_name,))
        concept_row = cursor.fetchone()
        
        if not concept_row:
            return {'attempts': 0, 'correct_count': 0, 'accuracy': 0.0}
            
        concept_id = concept_row[0]
        
        cursor.execute("""
            SELECT COUNT(*) as attempts,
                   SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct_count,
                   AVG(CASE WHEN correct = 1 THEN 1.0 ELSE 0.0 END) as accuracy
            FROM quiz_results
            WHERE concept_id = ?
        """, (concept_id,))
        
        row = cursor.fetchone()
        
        return {
            'attempts': row['attempts'] if row and row['attempts'] else 0,
            'correct_count': row['correct_count'] if row and row['correct_count'] else 0,
            'accuracy': row['accuracy'] if row and row['accuracy'] else 0.0
        }

    def update_concept_mastery(
        self,
        name: str,
        quiz_passed: Optional[bool] = None,
        implementation_passed: Optional[bool] = None,
        gpu_validated: Optional[bool] = None
    ):
        """Update concept mastery status"""
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
        """Get all concepts"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM concepts ORDER BY topic, name")
        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        """Close database connection"""
        self.conn.close()


class PostgreSQLBackend:
    """PostgreSQL backend for production progress tracking"""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS concepts (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) UNIQUE NOT NULL,
        topic VARCHAR(255) NOT NULL,
        difficulty VARCHAR(50) NOT NULL,
        quiz_passed BOOLEAN DEFAULT FALSE,
        implementation_passed BOOLEAN DEFAULT FALSE,
        gpu_validated BOOLEAN DEFAULT FALSE,
        last_reviewed DATE,
        next_review DATE,
        review_interval INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS quiz_results (
        id SERIAL PRIMARY KEY,
        concept_id INTEGER NOT NULL REFERENCES concepts(id),
        question_id VARCHAR(255) NOT NULL,
        correct BOOLEAN NOT NULL,
        timestamp TIMESTAMP DEFAULT NOW(),
        next_review DATE,
        review_interval INTEGER DEFAULT 1
    );

    CREATE INDEX IF NOT EXISTS idx_quiz_next_review ON quiz_results(next_review);

    CREATE TABLE IF NOT EXISTS gpu_jobs (
        id SERIAL PRIMARY KEY,
        job_id VARCHAR(255) UNIQUE NOT NULL,
        concept VARCHAR(255) NOT NULL,
        backend VARCHAR(100) NOT NULL,
        submitted_at TIMESTAMP NOT NULL,
        completed_at TIMESTAMP,
        status VARCHAR(50) NOT NULL,
        cost FLOAT DEFAULT 0.0,
        accuracy FLOAT,
        baseline_accuracy FLOAT,
        passed BOOLEAN DEFAULT FALSE
    );

    CREATE TABLE IF NOT EXISTS study_sessions (
        id SERIAL PRIMARY KEY,
        started_at TIMESTAMP NOT NULL,
        ended_at TIMESTAMP,
        concepts_studied TEXT,
        activities TEXT
    );
    """

    def __init__(self, database_url: str):
        """Initialize PostgreSQL connection

        Args:
            database_url: PostgreSQL connection URL
        """
        if not POSTGRES_AVAILABLE:
            raise ImportError("psycopg2 not available - install with: pip install psycopg2-binary")

        self.conn = psycopg2.connect(database_url)
        self._init_schema()

    def _init_schema(self):
        """Create database schema if not exists"""
        with self.conn.cursor() as cur:
            cur.execute(self.SCHEMA)
        self.conn.commit()

    def add_concept(self, name: str, topic: str, difficulty: str):
        """Add a new concept to track"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO concepts (name, topic, difficulty)
                VALUES (%s, %s, %s)
                ON CONFLICT (name) DO NOTHING
            """, (name, topic, difficulty))
        self.conn.commit()

    def get_concept(self, name: str) -> Optional[Dict[str, Any]]:
        """Get concept by name"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM concepts WHERE name=%s", (name,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_quiz_stats(self, concept_name: str) -> Dict[str, Any]:
        """Get aggregated quiz statistics for a concept"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get concept ID
            cur.execute("SELECT id FROM concepts WHERE name=%s", (concept_name,))
            concept_row = cur.fetchone()
            
            if not concept_row:
                return {'attempts': 0, 'correct_count': 0, 'accuracy': 0.0}
                
            concept_id = concept_row['id']
            
            cur.execute("""
                SELECT COUNT(*) as attempts,
                       SUM(CASE WHEN correct THEN 1 ELSE 0 END) as correct_count,
                       AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END) as accuracy
                FROM quiz_results
                WHERE concept_id = %s
            """, (concept_id,))
            
            row = cur.fetchone()
            
            return {
                'attempts': row['attempts'] if row and row['attempts'] else 0,
                'correct_count': row['correct_count'] if row and row['correct_count'] else 0,
                'accuracy': float(row['accuracy']) if row and row['accuracy'] else 0.0
            }

    def update_concept_mastery(
        self,
        name: str,
        quiz_passed: Optional[bool] = None,
        implementation_passed: Optional[bool] = None,
        gpu_validated: Optional[bool] = None
    ):
        """Update concept mastery status"""
        updates = []
        params = []

        if quiz_passed is not None:
            updates.append("quiz_passed = %s")
            params.append(quiz_passed)

        if implementation_passed is not None:
            updates.append("implementation_passed = %s")
            params.append(implementation_passed)

        if gpu_validated is not None:
            updates.append("gpu_validated = %s")
            params.append(gpu_validated)

        if not updates:
            return

        params.append(name)
        query = f"UPDATE concepts SET {', '.join(updates)} WHERE name=%s"

        with self.conn.cursor() as cur:
            cur.execute(query, params)
        self.conn.commit()

    def get_all_concepts(self) -> list:
        """Get all concepts"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM concepts ORDER BY topic, name")
            return [dict(row) for row in cur.fetchall()]

    def close(self):
        """Close database connection"""
        self.conn.close()


class ProgressDatabase:
    """Unified progress database with auto-detected backend

    Uses PostgreSQL if DATABASE_URL is set, otherwise falls back to SQLite.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database with appropriate backend

        Args:
            db_path: Path for SQLite database (ignored if DATABASE_URL is set)
        """
        database_url = os.environ.get('DATABASE_URL')

        if database_url and POSTGRES_AVAILABLE:
            self._backend = PostgreSQLBackend(database_url)
            self._backend_type = 'postgresql'
        else:
            if db_path is None:
                db_path = Path.home() / '.learning_framework' / 'progress.db'
            self._backend = SQLiteBackend(db_path)
            self._backend_type = 'sqlite'

    @property
    def backend_type(self) -> str:
        """Get current backend type ('sqlite' or 'postgresql')"""
        return self._backend_type

    def add_concept(self, name: str, topic: str, difficulty: str):
        """Add a new concept to track"""
        self._backend.add_concept(name, topic, difficulty)

    def get_concept(self, name: str) -> Optional[Dict[str, Any]]:
        """Get concept by name"""
        return self._backend.get_concept(name)

    def get_quiz_stats(self, concept_name: str) -> Dict[str, Any]:
        """Get aggregated quiz statistics"""
        return self._backend.get_quiz_stats(concept_name)

    def update_concept_mastery(
        self,
        name: str,
        quiz_passed: Optional[bool] = None,
        implementation_passed: Optional[bool] = None,
        gpu_validated: Optional[bool] = None
    ):
        """Update concept mastery status"""
        self._backend.update_concept_mastery(
            name, quiz_passed, implementation_passed, gpu_validated
        )

    def get_all_concepts(self) -> list:
        """Get all concepts"""
        return self._backend.get_all_concepts()

    def close(self):
        """Close database connection"""
        self._backend.close()