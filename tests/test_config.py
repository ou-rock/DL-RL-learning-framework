import pytest
import tempfile
import os
from pathlib import Path
from learning_framework.config import ConfigManager


def test_config_loads_default_values():
    """Test configuration loads with default values"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ConfigManager(config_dir=tmpdir)
        assert config.get('daily_gpu_budget') == 5.0
        assert config.get('max_job_cost') == 1.0
        assert config.get('editor') == 'code'


def test_config_loads_from_file():
    """Test configuration loads from YAML file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.yaml'
        config_path.write_text("""
daily_gpu_budget: 10.0
editor: vim
        """)

        config = ConfigManager(config_dir=tmpdir)
        assert config.get('daily_gpu_budget') == 10.0
        assert config.get('editor') == 'vim'
        assert config.get('max_job_cost') == 1.0  # Default value


def test_config_set_and_save():
    """Test setting and saving configuration"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ConfigManager(config_dir=tmpdir)
        config.set('vastai_api_key', 'test-key-123')
        config.save()

        # Load again to verify persistence
        config2 = ConfigManager(config_dir=tmpdir)
        assert config2.get('vastai_api_key') == 'test-key-123'


def test_config_get_nonexistent_returns_none():
    """Test getting nonexistent key returns None"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ConfigManager(config_dir=tmpdir)
        assert config.get('nonexistent_key') is None
