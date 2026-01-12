import pytest
from datetime import datetime, timedelta
import tempfile
from pathlib import Path
from learning_framework.assessment.spaced_repetition import SpacedRepetitionScheduler


def test_sm2_correct_answer_increases_interval():
    """Test SM-2: correct answer increases review interval"""
    with tempfile.TemporaryDirectory() as tmpdir:
        scheduler = SpacedRepetitionScheduler(db_path=Path(tmpdir) / 'test.db')

        try:
            # Correct answer: 1 day → ~3 days (1 * 2.5 = 2.5)
            next_review, new_interval = scheduler.calculate_next_review(
                item_id='test_001',
                correct=True,
                current_interval=1
            )

            assert new_interval == 2  # int(1 * 2.5) = 2
            assert (next_review - datetime.now()).days >= 1
        finally:
            scheduler.close()


def test_sm2_incorrect_answer_resets_interval():
    """Test SM-2: incorrect answer resets to day 1"""
    with tempfile.TemporaryDirectory() as tmpdir:
        scheduler = SpacedRepetitionScheduler(db_path=Path(tmpdir) / 'test.db')

        try:
            # Incorrect answer: any interval → 1 day
            next_review, new_interval = scheduler.calculate_next_review(
                item_id='test_001',
                correct=False,
                current_interval=14
            )

            assert new_interval == 1
            assert (next_review - datetime.now()).days == 0  # Tomorrow
        finally:
            scheduler.close()


def test_get_due_items_returns_items_due_today():
    """Test retrieving items due for review"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'test.db'

        # Setup: Create database with quiz results
        from learning_framework.progress import ProgressDatabase
        db = ProgressDatabase(db_path)
        db.add_concept('test_concept', 'test_topic', 'easy')

        # Add a quiz result that's due today
        concept = db.get_concept('test_concept')
        db.conn.execute("""
            INSERT INTO quiz_results
            (concept_id, question_id, correct, timestamp, next_review, review_interval)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (concept['id'], 'q_001', 1, datetime.now(),
              datetime.now().date(), 1))
        db.conn.commit()
        db.close()

        # Test: Get due items
        scheduler = SpacedRepetitionScheduler(db_path=db_path)
        try:
            due_items = scheduler.get_due_items(tier='quiz')

            assert len(due_items) > 0
            assert any(item['question_id'] == 'q_001' for item in due_items)
        finally:
            scheduler.close()
