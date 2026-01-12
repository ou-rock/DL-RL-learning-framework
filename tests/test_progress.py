import pytest
import tempfile
from pathlib import Path
from learning_framework.progress.database import ProgressDatabase


def test_database_initializes_schema():
    """Test database creates schema on initialization"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'progress.db'
        db = ProgressDatabase(db_path)

        # Verify tables exist
        cursor = db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        assert 'concepts' in tables
        assert 'quiz_results' in tables
        assert 'gpu_jobs' in tables
        assert 'study_sessions' in tables

        db.close()


def test_database_add_concept():
    """Test adding a concept to database"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = ProgressDatabase(Path(tmpdir) / 'progress.db')

        db.add_concept(
            name='backpropagation',
            topic='neural_networks',
            difficulty='intermediate'
        )

        cursor = db.conn.cursor()
        cursor.execute("SELECT name, topic, difficulty FROM concepts WHERE name=?",
                      ('backpropagation',))
        row = cursor.fetchone()

        assert row is not None
        assert row[0] == 'backpropagation'
        assert row[1] == 'neural_networks'
        assert row[2] == 'intermediate'

        db.close()


def test_database_get_concept():
    """Test retrieving a concept from database"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = ProgressDatabase(Path(tmpdir) / 'progress.db')

        db.add_concept('backpropagation', 'neural_networks', 'intermediate')
        concept = db.get_concept('backpropagation')

        assert concept is not None
        assert concept['name'] == 'backpropagation'
        assert concept['quiz_passed'] == 0
        assert concept['implementation_passed'] == 0
        assert concept['gpu_validated'] == 0

        db.close()


def test_database_update_concept_mastery():
    """Test updating concept mastery status"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = ProgressDatabase(Path(tmpdir) / 'progress.db')

        db.add_concept('backpropagation', 'neural_networks', 'intermediate')
        db.update_concept_mastery('backpropagation', quiz_passed=True)

        concept = db.get_concept('backpropagation')
        assert concept['quiz_passed'] == 1
        assert concept['implementation_passed'] == 0

        db.close()
