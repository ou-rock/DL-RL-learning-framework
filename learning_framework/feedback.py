"""Beta testing feedback collection system"""

import sqlite3
import json
from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import uuid
import platform


class FeedbackType(Enum):
    """Types of feedback"""
    BUG = "bug"
    FEATURE = "feature"
    USABILITY = "usability"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    OTHER = "other"


@dataclass
class Feedback:
    """Feedback entry"""
    id: str
    feedback_type: str
    title: str
    description: str
    steps_to_reproduce: Optional[str] = None
    severity: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    system_info: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    status: str = "open"


class FeedbackCollector:
    """Collects and manages user feedback for beta testing"""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize feedback collector

        Args:
            db_path: Path to feedback database
        """
        self.db_path = db_path or Path("user_data/feedback.db")
        self._init_database()

    def _init_database(self):
        """Initialize database schema"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    feedback_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    steps_to_reproduce TEXT,
                    severity TEXT,
                    context TEXT,
                    system_info TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'open'
                )
            ''')
            conn.commit()

    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information for debugging"""
        import sys
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "framework_version": self._get_version()
        }

    def _get_version(self) -> str:
        """Get framework version"""
        try:
            from learning_framework import __version__
            return __version__
        except ImportError:
            return "unknown"

    def submit(
        self,
        feedback_type: FeedbackType,
        title: str,
        description: str,
        steps_to_reproduce: Optional[str] = None,
        severity: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Submit new feedback

        Args:
            feedback_type: Type of feedback
            title: Short title
            description: Detailed description
            steps_to_reproduce: Steps to reproduce (for bugs)
            severity: Severity level (low, medium, high, critical)
            context: Additional context dict

        Returns:
            Feedback ID
        """
        feedback_id = f"FB-{uuid.uuid4().hex[:8].upper()}"
        system_info = self._get_system_info()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO feedback
                (id, feedback_type, title, description, steps_to_reproduce,
                 severity, context, system_info)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback_id,
                feedback_type.value,
                title,
                description,
                steps_to_reproduce,
                severity,
                json.dumps(context) if context else None,
                json.dumps(system_info)
            ))
            conn.commit()

        return feedback_id

    def list_feedback(
        self,
        feedback_type: Optional[FeedbackType] = None,
        status: Optional[str] = None
    ) -> List[Feedback]:
        """List submitted feedback

        Args:
            feedback_type: Filter by type
            status: Filter by status

        Returns:
            List of Feedback objects
        """
        query = "SELECT * FROM feedback WHERE 1=1"
        params = []

        if feedback_type:
            query += " AND feedback_type = ?"
            params.append(feedback_type.value)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

            return [
                Feedback(
                    id=row['id'],
                    feedback_type=row['feedback_type'],
                    title=row['title'],
                    description=row['description'],
                    steps_to_reproduce=row['steps_to_reproduce'],
                    severity=row['severity'],
                    context=json.loads(row['context']) if row['context'] else None,
                    system_info=json.loads(row['system_info']) if row['system_info'] else None,
                    created_at=row['created_at'],
                    status=row['status']
                )
                for row in rows
            ]

    def export(self, output_path: Path) -> None:
        """Export all feedback to JSON file

        Args:
            output_path: Path to output file
        """
        feedback_list = self.list_feedback()

        export_data = {
            "exported_at": datetime.now().isoformat(),
            "feedback_count": len(feedback_list),
            "feedback": [asdict(f) for f in feedback_list]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)

    def get_feedback(self, feedback_id: str) -> Optional[Feedback]:
        """Get specific feedback by ID

        Args:
            feedback_id: Feedback ID

        Returns:
            Feedback object or None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM feedback WHERE id = ?",
                (feedback_id,)
            ).fetchone()

            if row:
                return Feedback(
                    id=row['id'],
                    feedback_type=row['feedback_type'],
                    title=row['title'],
                    description=row['description'],
                    steps_to_reproduce=row['steps_to_reproduce'],
                    severity=row['severity'],
                    context=json.loads(row['context']) if row['context'] else None,
                    system_info=json.loads(row['system_info']) if row['system_info'] else None,
                    created_at=row['created_at'],
                    status=row['status']
                )
            return None
