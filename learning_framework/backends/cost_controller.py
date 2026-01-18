"""Cost tracking and budget enforcement for GPU jobs"""

import sqlite3
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Any, Optional


class CostController:
    """Tracks spending and enforces budget limits for GPU jobs"""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        daily_budget: float = 5.0,
        max_job_cost: float = 1.0
    ):
        """Initialize cost controller

        Args:
            db_path: Path to SQLite database (default: user_data/costs.db)
            daily_budget: Maximum daily spending in USD
            max_job_cost: Maximum cost per job in USD
        """
        self.db_path = db_path or Path("user_data/costs.db")
        self.daily_budget = daily_budget
        self.max_job_cost = max_job_cost

        self._init_database()

    def _init_database(self):
        """Initialize database schema"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS spending (
                    id INTEGER PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    amount REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    date TEXT NOT NULL,
                    description TEXT
                )
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_spending_date
                ON spending(date)
            ''')
            conn.commit()

    def record_spending(
        self,
        job_id: str,
        amount: float,
        description: str = ""
    ):
        """Record spending for a job

        Args:
            job_id: Job identifier
            amount: Amount spent in USD
            description: Optional description
        """
        today = date.today().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                '''INSERT INTO spending (job_id, amount, date, description)
                   VALUES (?, ?, ?, ?)''',
                (job_id, amount, today, description)
            )
            conn.commit()

    def get_daily_spending(self, target_date: Optional[date] = None) -> float:
        """Get total spending for a day

        Args:
            target_date: Date to check (default: today)

        Returns:
            Total spending in USD
        """
        if target_date is None:
            target_date = date.today()

        date_str = target_date.isoformat()

        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                'SELECT COALESCE(SUM(amount), 0) FROM spending WHERE date = ?',
                (date_str,)
            ).fetchone()
            return result[0]

    def can_spend(self, amount: float) -> bool:
        """Check if spending amount is allowed

        Args:
            amount: Proposed spending amount

        Returns:
            True if spending is allowed
        """
        # Check max job cost
        if amount > self.max_job_cost:
            return False

        # Check daily budget
        current_spending = self.get_daily_spending()
        if current_spending + amount > self.daily_budget:
            return False

        return True

    def get_remaining_budget(self) -> float:
        """Get remaining daily budget

        Returns:
            Remaining budget in USD
        """
        current = self.get_daily_spending()
        return max(0, self.daily_budget - current)

    def get_history(
        self,
        limit: int = 100,
        start_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """Get spending history

        Args:
            limit: Maximum records to return
            start_date: Start date filter (default: all)

        Returns:
            List of spending records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if start_date:
                query = '''
                    SELECT job_id, amount, timestamp, date, description
                    FROM spending
                    WHERE date >= ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                '''
                rows = conn.execute(query, (start_date.isoformat(), limit)).fetchall()
            else:
                query = '''
                    SELECT job_id, amount, timestamp, date, description
                    FROM spending
                    ORDER BY timestamp DESC
                    LIMIT ?
                '''
                rows = conn.execute(query, (limit,)).fetchall()

            return [dict(row) for row in rows]

    def get_job_cost(self, job_id: str) -> float:
        """Get total cost for a specific job

        Args:
            job_id: Job identifier

        Returns:
            Total cost for the job
        """
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                'SELECT COALESCE(SUM(amount), 0) FROM spending WHERE job_id = ?',
                (job_id,)
            ).fetchone()
            return result[0]
