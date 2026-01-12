2# Phase 1: Core Infrastructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build foundational infrastructure for the learning framework including project scaffolding, CLI, material indexer, progress database, and configuration management.

**Architecture:** Modular Python package with clear separation of concerns. CLI built with Click for command routing, SQLite for progress tracking, YAML for configuration. Material indexer uses convention-based scanning with manual annotation support.

**Tech Stack:** Python 3.10+, Click, Rich, SQLite3, PyYAML, pytest

---

## Task 1: Project Scaffolding

**Files:**
- Create: `learning-framework/` (new directory)
- Create: `learning-framework/setup.py`
- Create: `learning-framework/pyproject.toml`
- Create: `learning-framework/requirements.txt`
- Create: `learning-framework/requirements-dev.txt`
- Create: `learning-framework/README.md`
- Create: `learning-framework/.gitignore`

**Step 1: Create project directory**

```bash
mkdir learning-framework
cd learning-framework
```

**Step 2: Write setup.py**

Create `setup.py`:

```python
from setuptools import setup, find_packages

setup(
    name="learning-framework",
    version="0.1.0",
    description="Interactive DL/RL mastery framework",
    author="LH",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "click>=8.1.0",
        "rich>=13.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "paramiko>=3.3.0",
    ],
    entry_points={
        "console_scripts": [
            "lf=learning_framework.cli:cli",
        ],
    },
)
```

**Step 3: Write pyproject.toml**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "learning-framework"
version = "0.1.0"
description = "Interactive DL/RL mastery framework"
requires-python = ">=3.10"
dependencies = [
    "click>=8.1.0",
    "rich>=13.0.0",
    "numpy>=1.24.0",
    "matplotlib>=3.7.0",
    "pyyaml>=6.0",
    "requests>=2.31.0",
    "paramiko>=3.3.0",
]

[project.scripts]
lf = "learning_framework.cli:cli"
```

**Step 4: Write requirements.txt**

Create `requirements.txt`:

```
click>=8.1.0
rich>=13.0.0
numpy>=1.24.0
matplotlib>=3.7.0
pyyaml>=6.0
requests>=2.31.0
paramiko>=3.3.0
```

**Step 5: Write requirements-dev.txt**

Create `requirements-dev.txt`:

```
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
```

**Step 6: Write README.md**

Create `README.md`:

```markdown
# Learning Framework

Interactive Deep Learning & RL Mastery Framework

## Installation

```bash
pip install -e .
```

## Usage

```bash
lf --help
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
```

**Step 7: Write .gitignore**

Create `.gitignore`:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/

# User data
user_data/
materials/index.json

# Build
cpp/build/
*.o
*.so
*.dll
```

**Step 8: Initialize git repository**

```bash
git init
git add .
git commit -m "chore: initial project scaffolding"
```

---

## Task 2: Package Structure & Basic CLI

**Files:**
- Create: `learning_framework/__init__.py`
- Create: `learning_framework/cli.py`
- Create: `tests/__init__.py`
- Create: `tests/test_cli.py`

**Step 1: Write failing test for CLI entry point**

Create `tests/__init__.py` (empty file):

```python
# Test package
```

Create `tests/test_cli.py`:

```python
import pytest
from click.testing import CliRunner
from learning_framework.cli import cli


def test_cli_help():
    """Test CLI displays help message"""
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'Interactive Deep Learning' in result.output


def test_cli_version():
    """Test CLI displays version"""
    runner = CliRunner()
    result = runner.invoke(cli, ['--version'])
    assert result.exit_code == 0
    assert '0.1.0' in result.output
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_cli.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'learning_framework'"

**Step 3: Create package __init__.py**

Create `learning_framework/__init__.py`:

```python
"""Interactive DL/RL Mastery Framework"""

__version__ = "0.1.0"
```

**Step 4: Create basic CLI**

Create `learning_framework/cli.py`:

```python
"""Command-line interface for learning framework"""

import click
from rich.console import Console

from learning_framework import __version__

console = Console()


@click.group()
@click.version_option(version=__version__)
def cli():
    """Interactive Deep Learning & RL Mastery Framework

    Learn fundamentals locally, validate at scale on remote GPUs.
    """
    pass


@cli.command()
def learn():
    """Start interactive learning session"""
    console.print("[cyan]Learning session starting...[/cyan]")
    console.print("[yellow]Not yet implemented[/yellow]")


@cli.command()
def quiz():
    """Take concept quiz"""
    console.print("[cyan]Quiz starting...[/cyan]")
    console.print("[yellow]Not yet implemented[/yellow]")


@cli.command()
def progress():
    """View learning progress"""
    console.print("[cyan]Progress report...[/cyan]")
    console.print("[yellow]Not yet implemented[/yellow]")


@cli.command()
def index():
    """Re-index learning materials"""
    console.print("[cyan]Indexing materials...[/cyan]")
    console.print("[yellow]Not yet implemented[/yellow]")


@cli.command()
def config():
    """Configure framework settings"""
    console.print("[cyan]Configuration...[/cyan]")
    console.print("[yellow]Not yet implemented[/yellow]")


if __name__ == '__main__':
    cli()
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_cli.py -v
```

Expected: PASS (all tests pass)

**Step 6: Test CLI manually**

```bash
pip install -e .
lf --help
lf learn
```

Expected: Help message displays, learn command shows "Not yet implemented"

**Step 7: Commit**

```bash
git add learning_framework/ tests/
git commit -m "feat: add basic CLI with command placeholders"
```

---

## Task 3: Configuration Management

**Files:**
- Create: `learning_framework/config.py`
- Create: `tests/test_config.py`
- Create: `user_data/config-example.yaml`

**Step 1: Write failing test for configuration loading**

Create `tests/test_config.py`:

```python
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'learning_framework.config'"

**Step 3: Implement ConfigManager**

Create `learning_framework/config.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: PASS (all tests pass)

**Step 5: Create example config file**

```bash
mkdir -p user_data
```

Create `user_data/config-example.yaml`:

```yaml
# Learning Framework Configuration Example
# Copy this to user_data/config.yaml and customize

# GPU Backend Settings
vastai_api_key: "your-api-key-here"

# Budget Controls
daily_gpu_budget: 5.0      # Maximum daily spending on GPU (USD)
max_job_cost: 1.0          # Maximum cost per single job (USD)

# Editor & UI
editor: "code"             # Code editor (code, vim, nano, etc.)
auto_open_browser: true    # Auto-open browser for visualizations

# Learning Settings
quiz_questions_per_session: 10
spaced_repetition_enabled: true

# Materials
materials_directories:
  - "D:/ourock-test/ourock-test/DeepLearning"
  - "D:/ourock-test/ourock-test/RL"

# C++ Compilation
auto_compile: true         # Auto-compile C++ on first use
compiler: "auto"           # auto, msvc, gcc, clang
```

**Step 6: Integrate config into CLI**

Modify `learning_framework/cli.py` to add config command:

```python
"""Command-line interface for learning framework"""

import click
from rich.console import Console
from rich.table import Table

from learning_framework import __version__
from learning_framework.config import ConfigManager

console = Console()


@click.group()
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx):
    """Interactive Deep Learning & RL Mastery Framework

    Learn fundamentals locally, validate at scale on remote GPUs.
    """
    # Store config in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['config'] = ConfigManager()


@cli.command()
def learn():
    """Start interactive learning session"""
    console.print("[cyan]Learning session starting...[/cyan]")
    console.print("[yellow]Not yet implemented[/yellow]")


@cli.command()
def quiz():
    """Take concept quiz"""
    console.print("[cyan]Quiz starting...[/cyan]")
    console.print("[yellow]Not yet implemented[/yellow]")


@cli.command()
def progress():
    """View learning progress"""
    console.print("[cyan]Progress report...[/cyan]")
    console.print("[yellow]Not yet implemented[/yellow]")


@cli.command()
def index():
    """Re-index learning materials"""
    console.print("[cyan]Indexing materials...[/cyan]")
    console.print("[yellow]Not yet implemented[/yellow]")


@cli.command()
@click.pass_context
def config(ctx):
    """Configure framework settings"""
    config_mgr = ctx.obj['config']

    console.print("\n[bold cyan]Configuration[/bold cyan]\n")

    # Display current settings
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    all_config = config_mgr.get_all()
    for key, value in sorted(all_config.items()):
        # Mask API keys
        if 'key' in key.lower() and value:
            value = '*' * 8 + value[-4:] if len(str(value)) > 4 else '****'
        table.add_row(key, str(value))

    console.print(table)
    console.print(f"\n[dim]Config file: {config_mgr.config_path}[/dim]")
    console.print("[yellow]Use 'lf config --set key=value' to change settings (not yet implemented)[/yellow]")


if __name__ == '__main__':
    cli()
```

**Step 7: Test config command**

```bash
lf config
```

Expected: Displays configuration table with default values

**Step 8: Commit**

```bash
git add learning_framework/config.py tests/test_config.py user_data/
git commit -m "feat: add configuration management with YAML persistence"
```

---

## Task 4: Progress Database Schema

**Files:**
- Create: `learning_framework/progress/__init__.py`
- Create: `learning_framework/progress/database.py`
- Create: `learning_framework/progress/tracker.py`
- Create: `tests/test_progress.py`

**Step 1: Write failing test for database initialization**

Create `tests/test_progress.py`:

```python
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


def test_database_update_concept_mastery():
    """Test updating concept mastery status"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = ProgressDatabase(Path(tmpdir) / 'progress.db')

        db.add_concept('backpropagation', 'neural_networks', 'intermediate')
        db.update_concept_mastery('backpropagation', quiz_passed=True)

        concept = db.get_concept('backpropagation')
        assert concept['quiz_passed'] == 1
        assert concept['implementation_passed'] == 0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_progress.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'learning_framework.progress'"

**Step 3: Create progress package __init__.py**

Create `learning_framework/progress/__init__.py`:

```python
"""Progress tracking and database management"""

from learning_framework.progress.database import ProgressDatabase
from learning_framework.progress.tracker import ProgressTracker

__all__ = ['ProgressDatabase', 'ProgressTracker']
```

**Step 4: Implement ProgressDatabase**

Create `learning_framework/progress/database.py`:

```python
"""SQLite database for progress tracking"""

import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class ProgressDatabase:
    """Manages SQLite database for learning progress"""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS concepts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        topic TEXT NOT NULL,
        difficulty TEXT NOT NULL,
        quiz_passed BOOLEAN DEFAULT 0,
        implementation_passed BOOLEAN DEFAULT 0,
        gpu_validated BOOLEAN DEFAULT 0,
        last_reviewed DATE,
        next_review DATE,
        review_interval INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS quiz_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        concept_id INTEGER NOT NULL,
        question_id TEXT NOT NULL,
        correct BOOLEAN NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (concept_id) REFERENCES concepts(id)
    );

    CREATE TABLE IF NOT EXISTS gpu_jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT UNIQUE NOT NULL,
        concept TEXT NOT NULL,
        backend TEXT NOT NULL,
        submitted_at DATETIME NOT NULL,
        completed_at DATETIME,
        status TEXT NOT NULL,
        cost REAL DEFAULT 0.0,
        accuracy REAL,
        baseline_accuracy REAL,
        passed BOOLEAN DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS study_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at DATETIME NOT NULL,
        ended_at DATETIME,
        concepts_studied TEXT,
        activities TEXT
    );
    """

    def __init__(self, db_path: Path):
        """Initialize database connection and schema

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        self._init_schema()

    def _init_schema(self):
        """Create database schema if not exists"""
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def add_concept(self, name: str, topic: str, difficulty: str):
        """Add a new concept to track

        Args:
            name: Concept name (unique identifier)
            topic: Topic category
            difficulty: Difficulty level (easy, intermediate, hard)
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO concepts (name, topic, difficulty)
            VALUES (?, ?, ?)
        """, (name, topic, difficulty))
        self.conn.commit()

    def get_concept(self, name: str) -> Optional[Dict[str, Any]]:
        """Get concept by name

        Args:
            name: Concept name

        Returns:
            Concept data as dictionary or None
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM concepts WHERE name=?", (name,))
        row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    def update_concept_mastery(
        self,
        name: str,
        quiz_passed: Optional[bool] = None,
        implementation_passed: Optional[bool] = None,
        gpu_validated: Optional[bool] = None
    ):
        """Update concept mastery status

        Args:
            name: Concept name
            quiz_passed: Quiz completion status
            implementation_passed: Implementation completion status
            gpu_validated: GPU validation status
        """
        updates = []
        params = []

        if quiz_passed is not None:
            updates.append("quiz_passed = ?")
            params.append(int(quiz_passed))

        if implementation_passed is not None:
            updates.append("implementation_passed = ?")
            params.append(int(implementation_passed))

        if gpu_validated is not None:
            updates.append("gpu_validated = ?")
            params.append(int(gpu_validated))

        if not updates:
            return

        params.append(name)
        query = f"UPDATE concepts SET {', '.join(updates)} WHERE name=?"

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        self.conn.commit()

    def get_all_concepts(self) -> list:
        """Get all concepts

        Returns:
            List of concept dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM concepts ORDER BY topic, name")
        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        """Close database connection"""
        self.conn.close()
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_progress.py -v
```

Expected: PASS (all tests pass)

**Step 6: Implement ProgressTracker wrapper**

Create `learning_framework/progress/tracker.py`:

```python
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
```

**Step 7: Commit**

```bash
git add learning_framework/progress/ tests/test_progress.py
git commit -m "feat: add progress database with SQLite schema"
```

---

## Task 5: Material Indexer Foundation

**Files:**
- Create: `learning_framework/knowledge/__init__.py`
- Create: `learning_framework/knowledge/indexer.py`
- Create: `tests/test_indexer.py`

**Step 1: Write failing test for directory scanning**

Create `tests/test_indexer.py`:

```python
import pytest
import tempfile
from pathlib import Path
from learning_framework.knowledge.indexer import MaterialIndexer


def test_indexer_scans_directories():
    """Test indexer scans directories and finds files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test directory structure
        base = Path(tmpdir)
        (base / 'ch01').mkdir()
        (base / 'ch01' / 'README.md').write_text('# Chapter 1: Introduction')
        (base / 'ch01' / 'example.py').write_text('# Example code')
        (base / 'ch02').mkdir()
        (base / 'ch02' / 'README.md').write_text('# Chapter 2: Basics')

        indexer = MaterialIndexer()
        results = indexer.scan_directory(base)

        assert len(results['chapters']) == 2
        assert 'ch01' in results['chapters']
        assert 'ch02' in results['chapters']
        assert len(results['files']) >= 3


def test_indexer_detects_chapter_pattern():
    """Test indexer detects chapter naming patterns"""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        (base / 'ch01').mkdir()
        (base / 'ch02').mkdir()
        (base / 'chapter03').mkdir()
        (base / 'other').mkdir()

        indexer = MaterialIndexer()
        results = indexer.scan_directory(base)

        chapters = results['chapters']
        assert 'ch01' in chapters
        assert 'ch02' in chapters
        assert 'chapter03' in chapters
        assert 'other' not in chapters


def test_indexer_finds_python_files():
    """Test indexer finds Python files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        (base / 'script.py').write_text('print("hello")')
        (base / 'train.py').write_text('# training script')
        (base / 'README.md').write_text('# Readme')

        indexer = MaterialIndexer()
        results = indexer.scan_directory(base)

        py_files = [f for f in results['files'] if f.endswith('.py')]
        assert len(py_files) == 2


def test_indexer_extracts_ml_keywords():
    """Test indexer extracts ML keywords from Python files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        # File with ML keywords
        (base / 'backprop.py').write_text('''
def backpropagation(network, loss):
    """Compute gradients via backprop"""
    gradients = compute_gradients(loss)
    return gradients
        ''')

        indexer = MaterialIndexer()
        results = indexer.scan_directory(base)

        keywords = results.get('keywords', [])
        assert 'backpropagation' in keywords or 'backprop' in keywords
        assert 'gradients' in keywords or 'gradient' in keywords
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_indexer.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create knowledge package**

Create `learning_framework/knowledge/__init__.py`:

```python
"""Knowledge graph and material indexing"""

from learning_framework.knowledge.indexer import MaterialIndexer

__all__ = ['MaterialIndexer']
```

**Step 4: Implement MaterialIndexer**

Create `learning_framework/knowledge/indexer.py`:

```python
"""Material indexer for discovering learning resources"""

import re
from pathlib import Path
from typing import Dict, List, Any


class MaterialIndexer:
    """Scans directories to discover and index learning materials"""

    # Common ML/DL keywords to detect
    ML_KEYWORDS = [
        'backprop', 'gradient', 'loss', 'optimization', 'optimizer',
        'neural', 'network', 'layer', 'activation', 'sigmoid', 'relu',
        'convolution', 'pooling', 'dropout', 'batch_norm',
        'q_learning', 'policy', 'reward', 'agent', 'environment',
        'reinforcement', 'dqn', 'actor', 'critic'
    ]

    # Patterns for chapter detection
    CHAPTER_PATTERNS = [
        r'^ch\d+$',          # ch01, ch02, etc.
        r'^chapter\d+$',     # chapter01, chapter02
        r'^step\d+$',        # step01, step02 (for dezero)
    ]

    def scan_directory(self, root_path: Path) -> Dict[str, Any]:
        """Scan directory for learning materials

        Args:
            root_path: Root directory to scan

        Returns:
            Dictionary with scan results
        """
        root_path = Path(root_path)

        results = {
            'root': str(root_path),
            'chapters': [],
            'files': [],
            'keywords': set(),
        }

        # Scan directory structure
        for item in root_path.rglob('*'):
            if item.is_dir():
                # Check for chapter pattern
                if self._is_chapter_dir(item.name):
                    rel_path = item.relative_to(root_path)
                    results['chapters'].append(str(rel_path))

            elif item.is_file():
                rel_path = item.relative_to(root_path)
                results['files'].append(str(rel_path))

                # Extract keywords from Python files
                if item.suffix == '.py':
                    keywords = self._extract_keywords(item)
                    results['keywords'].update(keywords)

        # Convert set to list for JSON serialization
        results['keywords'] = list(results['keywords'])

        return results

    def _is_chapter_dir(self, dirname: str) -> bool:
        """Check if directory name matches chapter pattern

        Args:
            dirname: Directory name

        Returns:
            True if matches chapter pattern
        """
        dirname_lower = dirname.lower()
        return any(
            re.match(pattern, dirname_lower)
            for pattern in self.CHAPTER_PATTERNS
        )

    def _extract_keywords(self, file_path: Path) -> List[str]:
        """Extract ML keywords from Python file

        Args:
            file_path: Path to Python file

        Returns:
            List of found keywords
        """
        try:
            content = file_path.read_text(encoding='utf-8').lower()
        except Exception:
            return []

        found_keywords = []
        for keyword in self.ML_KEYWORDS:
            if keyword in content:
                found_keywords.append(keyword)

        return found_keywords
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_indexer.py -v
```

Expected: PASS (all tests pass)

**Step 6: Integrate indexer with CLI**

Update `learning_framework/cli.py` to add working index command:

```python
# ... existing imports ...
from learning_framework.knowledge import MaterialIndexer
from learning_framework.config import ConfigManager
import json

# ... existing code ...

@cli.command()
@click.pass_context
def index(ctx):
    """Re-index learning materials"""
    config_mgr = ctx.obj['config']

    console.print("[cyan]Indexing materials...[/cyan]\n")

    materials_dirs = config_mgr.get('materials_directories', [])

    if not materials_dirs:
        console.print("[yellow]No materials directories configured.[/yellow]")
        console.print("[dim]Add directories in user_data/config.yaml[/dim]")
        return

    indexer = MaterialIndexer()
    all_results = {}

    for mat_dir in materials_dirs:
        mat_path = Path(mat_dir)
        if not mat_path.exists():
            console.print(f"[red]Directory not found: {mat_dir}[/red]")
            continue

        console.print(f"Scanning: {mat_dir}")
        results = indexer.scan_directory(mat_path)
        all_results[str(mat_path)] = results

        console.print(f"  Found: {len(results['chapters'])} chapters, "
                     f"{len(results['files'])} files, "
                     f"{len(results['keywords'])} keywords")

    # Save index
    index_path = Path('materials') / 'index.json'
    index_path.parent.mkdir(exist_ok=True)

    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    console.print(f"\n[green]✓ Index saved to: {index_path}[/green]")
```

Add import at top of cli.py:

```python
from pathlib import Path
import json
```

**Step 7: Test indexer CLI**

```bash
lf index
```

Expected: If materials directories configured, scans and creates index.json

**Step 8: Commit**

```bash
git add learning_framework/knowledge/ tests/test_indexer.py learning_framework/cli.py
git commit -m "feat: add material indexer with chapter detection"
```

---

## Task 6: Integration Testing & Documentation

**Files:**
- Create: `tests/test_integration.py`
- Modify: `README.md`

**Step 1: Write integration test**

Create `tests/test_integration.py`:

```python
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
```

**Step 2: Run integration tests**

```bash
pytest tests/test_integration.py -v
```

Expected: PASS (all tests pass)

**Step 3: Update README with usage instructions**

Update `README.md`:

```markdown
# Learning Framework

Interactive Deep Learning & RL Mastery Framework

Learn DL/RL fundamentals with minimal resources, validate at scale on remote GPUs.

## Features

- **Progressive Mastery**: Quiz → Implementation → GPU Validation
- **Minimal Resources**: Learn on CPU, scale to GPU when ready
- **Material Indexing**: Auto-discover your existing learning materials
- **Progress Tracking**: SQLite-based progress database with spaced repetition
- **Multi-Backend GPU**: Support for Vast.ai, Colab, SSH servers

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Install

```bash
git clone <repository-url>
cd learning-framework
pip install -e .
```

### Development Installation

```bash
pip install -e .
pip install -r requirements-dev.txt
```

## Quick Start

### 1. Configure

Create `user_data/config.yaml` (copy from `user_data/config-example.yaml`):

```yaml
materials_directories:
  - "/path/to/your/DeepLearning"
  - "/path/to/your/RL"

daily_gpu_budget: 5.0
editor: "code"
```

### 2. Index Materials

```bash
lf index
```

### 3. View Progress

```bash
lf progress
```

### 4. Start Learning (coming soon)

```bash
lf learn
```

## Available Commands

```bash
lf --help              # Show all commands
lf config              # View configuration
lf index               # Index learning materials
lf progress            # View learning progress
lf learn               # Start learning (coming soon)
lf quiz                # Take quiz (coming soon)
```

## Project Structure

```
learning-framework/
├── learning_framework/      # Main package
│   ├── cli.py              # CLI interface
│   ├── config.py           # Configuration management
│   ├── knowledge/          # Material indexer
│   └── progress/           # Progress tracking
├── tests/                  # Test suite
├── user_data/              # User configuration and data
└── materials/              # Indexed materials (auto-generated)
```

## Development

### Run Tests

```bash
pytest
pytest -v                    # Verbose
pytest --cov                 # With coverage
```

### Code Style

```bash
black learning_framework/ tests/
flake8 learning_framework/ tests/
```

## Phase 1 Complete ✓

- [x] Project scaffolding
- [x] CLI framework with Click + Rich
- [x] Configuration management (YAML)
- [x] Progress database (SQLite)
- [x] Material indexer (auto-discovery)

## Coming Soon

- Phase 2: Quiz system with spaced repetition
- Phase 3: Implementation challenges
- Phase 4: Interactive visualizations
- Phase 5: GPU backend (Vast.ai)
- Phase 6: C++ implementations
```

**Step 4: Run all tests**

```bash
pytest -v
```

Expected: All tests pass

**Step 5: Final commit**

```bash
git add README.md tests/test_integration.py
git commit -m "docs: update README with Phase 1 completion"
```

---

## Task 7: Package Distribution Setup

**Files:**
- Create: `MANIFEST.in`
- Create: `.github/workflows/tests.yml` (optional, for CI)

**Step 1: Create MANIFEST.in for package data**

Create `MANIFEST.in`:

```
include README.md
include requirements.txt
include requirements-dev.txt
recursive-include user_data *.yaml
recursive-include data *.json
```

**Step 2: Test package installation**

```bash
pip install -e .
lf --version
```

Expected: Shows version 0.1.0

**Step 3: Create basic CI workflow (optional)**

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -e .
        pip install -r requirements-dev.txt

    - name: Run tests
      run: pytest -v --cov=learning_framework

    - name: Check code style
      run: |
        black --check learning_framework/ tests/
        flake8 learning_framework/ tests/
```

**Step 4: Final commit**

```bash
git add MANIFEST.in .github/
git commit -m "build: add package distribution and CI setup"
```

**Step 5: Create git tag for Phase 1**

```bash
git tag -a v0.1.0-phase1 -m "Phase 1: Core Infrastructure Complete"
```

---

## Phase 1 Completion Checklist

Run this verification checklist:

```bash
# 1. All tests pass
pytest -v

# 2. CLI works
lf --help
lf --version
lf config
lf index
lf progress

# 3. Package installs cleanly
pip install -e .

# 4. Code quality
black --check learning_framework/ tests/
flake8 learning_framework/ tests/

# 5. Git history clean
git log --oneline

# 6. Documentation complete
cat README.md
```

Expected results:
- ✓ All tests passing
- ✓ All CLI commands execute without errors
- ✓ Package installs successfully
- ✓ Code follows style guidelines
- ✓ Git history shows logical commits
- ✓ README accurately reflects current state

---

## Next Steps

**Phase 1 is complete!** You now have:
- ✅ Project scaffolding with proper Python package structure
- ✅ CLI framework using Click + Rich
- ✅ Configuration management with YAML persistence
- ✅ Progress tracking with SQLite database
- ✅ Material indexer with auto-discovery
- ✅ Comprehensive test suite
- ✅ Documentation

**Ready for Phase 2:** Learning & Assessment
- Knowledge graph implementation
- Quiz system extending vocab quiz architecture
- Spaced repetition algorithm (SM-2)
- Static visualizations with matplotlib
- Concept database for core DL/RL topics

---

## Execution Notes

**Estimated Time:** 4-6 hours for complete Phase 1 implementation

**Key Principles Applied:**
- **TDD**: All features test-driven
- **DRY**: Reusable components (ConfigManager, ProgressDatabase)
- **YAGNI**: Only Phase 1 features, no premature optimization
- **Frequent commits**: Logical checkpoints for easy rollback

**Dependencies Installed:**
- click (CLI framework)
- rich (terminal formatting)
- pyyaml (configuration)
- pytest (testing)
- numpy, matplotlib (future use)

**Files Created:** 25+
**Tests Written:** 15+
**Git Commits:** 7 major commits
