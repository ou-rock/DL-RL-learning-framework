"""High-level progress tracking interface"""

from pathlib import Path
from typing import Optional, Dict, Any
from learning_framework.progress.database import ProgressDatabase


class ProgressTracker:
    """High-level interface for tracking learning progress"""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize progress tracker

        Args:
            db_path: Path to database (default: user_data/progress.db)
        """
        if db_path is None:
            db_path = Path.cwd() / 'user_data' / 'progress.db'

        self.db = ProgressDatabase(db_path)

    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall progress statistics

        Returns:
            Dictionary with overall stats
        """
        concepts = self.db.get_all_concepts()

        total = len(concepts)
        mastered = sum(
            1 for c in concepts
            if c['quiz_passed'] and c['implementation_passed'] and c['gpu_validated']
        )

        return {
            'total': total,
            'mastered': mastered,
            'in_progress': total - mastered,
            'quiz_passed': sum(1 for c in concepts if c['quiz_passed']),
            'implementation_passed': sum(1 for c in concepts if c['implementation_passed']),
            'gpu_validated': sum(1 for c in concepts if c['gpu_validated']),
        }

    def get_topic_stats(self, topic: str) -> Dict[str, Any]:
        """Get statistics for specific topic

        Args:
            topic: Topic name

        Returns:
            Dictionary with topic stats
        """
        concepts = [c for c in self.db.get_all_concepts() if c['topic'] == topic]

        total = len(concepts)
        mastered = sum(
            1 for c in concepts
            if c['quiz_passed'] and c['implementation_passed'] and c['gpu_validated']
        )

        return {
            'topic': topic,
            'total': total,
            'mastered': mastered,
            'concepts': concepts,
        }

    def close(self):
        """Close database connection"""
        self.db.close()
