"""Integration tests for core infrastructure"""

import pytest
import tempfile
from pathlib import Path
from click.testing import CliRunner
from learning_framework.cli import cli


def test_full_workflow():
    """Test complete workflow: config -> index -> progress"""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = CliRunner()

        # Test CLI loads
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0

        # Test config command
        result = runner.invoke(cli, ['config'])
        assert result.exit_code == 0
        assert 'Configuration' in result.output

        # Test progress command
        result = runner.invoke(cli, ['progress'])
        assert result.exit_code == 0


def test_config_persistence():
    """Test configuration persists across invocations"""
    with tempfile.TemporaryDirectory() as tmpdir:
        from learning_framework.config import ConfigManager

        # Set value
        config1 = ConfigManager(config_dir=tmpdir)
        config1.set('test_key', 'test_value')
        config1.save()

        # Load in new instance
        config2 = ConfigManager(config_dir=tmpdir)
        assert config2.get('test_key') == 'test_value'


def test_database_persistence():
    """Test database persists data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        from learning_framework.progress import ProgressDatabase

        db_path = Path(tmpdir) / 'test.db'

        # Add concept
        db1 = ProgressDatabase(db_path)
        db1.add_concept('test_concept', 'test_topic', 'easy')
        db1.close()

        # Verify in new connection
        db2 = ProgressDatabase(db_path)
        concept = db2.get_concept('test_concept')
        assert concept is not None
        assert concept['name'] == 'test_concept'
        db2.close()
