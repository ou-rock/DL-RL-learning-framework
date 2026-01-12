"""Spaced repetition scheduler using SM-2 algorithm

Extracted from vocab quiz system for reusability across
both vocabulary learning and ML concept quizzes.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from learning_framework.progress import ProgressDatabase


class SpacedRepetitionScheduler:
    """SM-2 spaced repetition algorithm for optimal review scheduling

    Works for any learning items: German vocab, ML concepts, etc.

    Algorithm:
    - Correct answer: interval *= 2.5
    - Incorrect answer: reset to 1 day
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize scheduler

        Args:
            db_path: Path to progress database (default: user_data/progress.db)
        """
        if db_path is None:
            db_path = Path.cwd() / 'user_data' / 'progress.db'

        self.db = ProgressDatabase(db_path)

    def calculate_next_review(
        self,
        item_id: str,
        correct: bool,
        current_interval: int = 1
    ) -> Tuple[datetime, int]:
        """Calculate next review date using SM-2 algorithm

        Args:
            item_id: Question ID or item identifier
            correct: Whether answer was correct
            current_interval: Current interval in days

        Returns:
            (next_review_date, new_interval_days)
        """
        if correct:
            # Correct answer: increase interval
            new_interval = max(1, int(current_interval * 2.5))
        else:
            # Incorrect answer: reset to day 1
            new_interval = 1

        # Calculate next review date
        next_review = datetime.now() + timedelta(days=new_interval)

        return next_review, new_interval

    def get_due_items(
        self,
        concept: Optional[str] = None,
        tier: str = 'quiz'
    ) -> List[Dict[str, Any]]:
        """Get items due for review today

        Args:
            concept: Filter by concept name (optional)
            tier: 'quiz' or 'implementation'

        Returns:
            List of due items with metadata
        """
        query = """
            SELECT
                qr.id,
                qr.question_id,
                qr.next_review,
                qr.review_interval,
                c.name as concept_name,
                c.topic as concept_topic
            FROM quiz_results qr
            JOIN concepts c ON qr.concept_id = c.id
            WHERE qr.next_review <= DATE('now')
        """

        params = []

        if concept:
            query += " AND c.name = ?"
            params.append(concept)

        query += " ORDER BY qr.next_review ASC"

        cursor = self.db.conn.cursor()
        cursor.execute(query, params)

        results = []
        for row in cursor.fetchall():
            results.append(dict(row))

        return results

    def close(self):
        """Close database connection"""
        self.db.close()
