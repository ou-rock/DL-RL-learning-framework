"""Database models and repository classes for user progress tracking"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any
from .connection import db_connection


# SQL Schema
SCHEMA_SQL = """
-- Users table (anonymous users identified by UUID)
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    user_id UUID UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Learning progress table
CREATE TABLE IF NOT EXISTS progress (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    concept VARCHAR(100) NOT NULL,
    level INTEGER DEFAULT 0,
    total_score FLOAT DEFAULT 0,
    attempts INTEGER DEFAULT 0,
    last_activity TIMESTAMP,
    UNIQUE(user_id, concept)
);

-- Spaced repetition reviews table
CREATE TABLE IF NOT EXISTS reviews (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    concept VARCHAR(100) NOT NULL,
    next_review DATE,
    interval_days INTEGER DEFAULT 1,
    ease_factor FLOAT DEFAULT 2.5,
    UNIQUE(user_id, concept)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_progress_user ON progress(user_id);
CREATE INDEX IF NOT EXISTS idx_reviews_user_date ON reviews(user_id, next_review);
"""


def ensure_schema() -> bool:
    """Create database tables if they don't exist

    Returns:
        True if schema created/verified successfully
    """
    with db_connection() as conn:
        if conn is None:
            return False
        try:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
            return True
        except Exception as e:
            print(f"Failed to create schema: {e}")
            return False


class UserRepository:
    """Repository for user management"""

    @staticmethod
    def ensure_user(user_id: str) -> bool:
        """Ensure user exists, create if not

        Args:
            user_id: UUID string from frontend

        Returns:
            True if user exists or was created
        """
        with db_connection() as conn:
            if conn is None:
                return False
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO users (user_id)
                        VALUES (%s)
                        ON CONFLICT (user_id) DO NOTHING
                        """,
                        (user_id,)
                    )
                return True
            except Exception as e:
                print(f"Failed to ensure user: {e}")
                return False

    @staticmethod
    def get_user(user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by UUID"""
        with db_connection() as conn:
            if conn is None:
                return None
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, user_id, created_at FROM users WHERE user_id = %s",
                        (user_id,)
                    )
                    row = cur.fetchone()
                    if row:
                        return {
                            'id': row[0],
                            'user_id': str(row[1]),
                            'created_at': row[2].isoformat() if row[2] else None
                        }
                return None
            except Exception as e:
                print(f"Failed to get user: {e}")
                return None


class ProgressRepository:
    """Repository for learning progress tracking"""

    @staticmethod
    def get_progress(user_id: str, concept: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get progress for user, optionally filtered by concept

        Args:
            user_id: User UUID
            concept: Optional concept slug to filter by

        Returns:
            List of progress records
        """
        with db_connection() as conn:
            if conn is None:
                return []
            try:
                with conn.cursor() as cur:
                    if concept:
                        cur.execute(
                            """
                            SELECT concept, level, total_score, attempts, last_activity
                            FROM progress WHERE user_id = %s AND concept = %s
                            """,
                            (user_id, concept)
                        )
                    else:
                        cur.execute(
                            """
                            SELECT concept, level, total_score, attempts, last_activity
                            FROM progress WHERE user_id = %s
                            ORDER BY last_activity DESC NULLS LAST
                            """,
                            (user_id,)
                        )
                    rows = cur.fetchall()
                    return [
                        {
                            'concept': row[0],
                            'level': row[1],
                            'total_score': row[2],
                            'attempts': row[3],
                            'last_activity': row[4].isoformat() if row[4] else None
                        }
                        for row in rows
                    ]
            except Exception as e:
                print(f"Failed to get progress: {e}")
                return []

    @staticmethod
    def update_progress(user_id: str, concept: str, score: float) -> bool:
        """Update or create progress record

        Args:
            user_id: User UUID
            concept: Concept slug
            score: Quiz score (0.0 - 1.0)

        Returns:
            True if updated successfully
        """
        with db_connection() as conn:
            if conn is None:
                return False
            try:
                with conn.cursor() as cur:
                    # Upsert progress record
                    cur.execute(
                        """
                        INSERT INTO progress (user_id, concept, level, total_score, attempts, last_activity)
                        VALUES (%s, %s, %s, %s, 1, NOW())
                        ON CONFLICT (user_id, concept) DO UPDATE SET
                            level = CASE
                                WHEN EXCLUDED.total_score >= 0.8 THEN LEAST(progress.level + 1, 3)
                                WHEN EXCLUDED.total_score < 0.5 THEN GREATEST(progress.level - 1, 0)
                                ELSE progress.level
                            END,
                            total_score = (progress.total_score * progress.attempts + EXCLUDED.total_score) / (progress.attempts + 1),
                            attempts = progress.attempts + 1,
                            last_activity = NOW()
                        """,
                        (user_id, concept, 1 if score >= 0.8 else 0, score)
                    )
                return True
            except Exception as e:
                print(f"Failed to update progress: {e}")
                return False


class ReviewRepository:
    """Repository for spaced repetition scheduling"""

    @staticmethod
    def get_due_reviews(user_id: str) -> List[Dict[str, Any]]:
        """Get concepts due for review

        Args:
            user_id: User UUID

        Returns:
            List of concepts due for review today or earlier
        """
        with db_connection() as conn:
            if conn is None:
                return []
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT concept, next_review, interval_days, ease_factor
                        FROM reviews
                        WHERE user_id = %s AND next_review <= CURRENT_DATE
                        ORDER BY next_review ASC
                        """,
                        (user_id,)
                    )
                    rows = cur.fetchall()
                    return [
                        {
                            'concept': row[0],
                            'next_review': row[1].isoformat() if row[1] else None,
                            'interval_days': row[2],
                            'ease_factor': row[3]
                        }
                        for row in rows
                    ]
            except Exception as e:
                print(f"Failed to get due reviews: {e}")
                return []

    @staticmethod
    def schedule_review(user_id: str, concept: str, quality: int) -> bool:
        """Schedule next review using SM-2 algorithm

        Args:
            user_id: User UUID
            concept: Concept slug
            quality: Response quality (0-5, where 5 is perfect recall)

        Returns:
            True if scheduled successfully
        """
        with db_connection() as conn:
            if conn is None:
                return False
            try:
                with conn.cursor() as cur:
                    # Get current review state
                    cur.execute(
                        """
                        SELECT interval_days, ease_factor FROM reviews
                        WHERE user_id = %s AND concept = %s
                        """,
                        (user_id, concept)
                    )
                    row = cur.fetchone()

                    if row:
                        interval, ease = row
                    else:
                        interval, ease = 1, 2.5

                    # SM-2 algorithm
                    if quality >= 3:
                        if interval == 1:
                            new_interval = 1
                        elif interval == 2:
                            new_interval = 6
                        else:
                            new_interval = int(interval * ease)
                        new_ease = ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
                    else:
                        new_interval = 1
                        new_ease = ease

                    new_ease = max(1.3, new_ease)

                    # Upsert review record
                    cur.execute(
                        """
                        INSERT INTO reviews (user_id, concept, next_review, interval_days, ease_factor)
                        VALUES (%s, %s, CURRENT_DATE + %s, %s, %s)
                        ON CONFLICT (user_id, concept) DO UPDATE SET
                            next_review = CURRENT_DATE + EXCLUDED.interval_days,
                            interval_days = EXCLUDED.interval_days,
                            ease_factor = EXCLUDED.ease_factor
                        """,
                        (user_id, concept, new_interval, new_interval, new_ease)
                    )
                return True
            except Exception as e:
                print(f"Failed to schedule review: {e}")
                return False
