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
