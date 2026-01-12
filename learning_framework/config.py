"""Configuration management for learning framework"""

import os
from pathlib import Path
from typing import Any, Optional
import yaml


class ConfigManager:
    """Manages framework configuration with YAML persistence"""

    DEFAULT_CONFIG = {
        'daily_gpu_budget': 5.0,
        'max_job_cost': 1.0,
        'editor': 'code',
        'auto_open_browser': True,
        'quiz_questions_per_session': 10,
        'spaced_repetition_enabled': True,
        'auto_compile': True,
        'compiler': 'auto',
        'materials_directories': [],
    }

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager

        Args:
            config_dir: Directory for config file (default: user_data/)
        """
        if config_dir is None:
            config_dir = Path.cwd() / 'user_data'

        self.config_dir = Path(config_dir)
        self.config_path = self.config_dir / 'config.yaml'
        self._config = self.DEFAULT_CONFIG.copy()

        # Create directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Load existing config if available
        if self.config_path.exists():
            self._load()

    def _load(self):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f) or {}
        self._config.update(user_config)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        """Set configuration value

        Args:
            key: Configuration key
            value: Value to set
        """
        self._config[key] = value

    def save(self):
        """Save configuration to YAML file"""
        # Only save non-default values to keep config clean
        save_config = {}
        for key, value in self._config.items():
            if key not in self.DEFAULT_CONFIG or self.DEFAULT_CONFIG[key] != value:
                save_config[key] = value

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(save_config, f, default_flow_style=False, allow_unicode=True)

    def get_all(self) -> dict:
        """Get all configuration values

        Returns:
            Dictionary of all configuration
        """
        return self._config.copy()
