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
lf challenge --list    # List implementation challenges
lf challenge <name>    # Start a challenge
lf test <name>         # Test your implementation
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

## Phase 2 Complete ✓

- [x] Quiz system for Tier 1 assessment
- [x] Spaced repetition scheduler
- [x] Concept tracking

## Phase 3 Complete ✓

- [x] Challenge template system (fill, scratch, debug)
- [x] Automated test runner with pytest
- [x] Numerical gradient checking
- [x] Backprop fill-in-blank challenge
- [x] SGD from-scratch challenge
- [x] CLI commands for challenges and testing

See [Phase 3 Usage Guide](docs/PHASE3_USAGE.md) for details.

## Phase 4 Complete ✓

- [x] Interactive visualization server
- [x] Concept graph visualization
- [x] Parameter explorer
- [x] Training monitor
- [x] Real-time data updates

See [Phase 4 Usage Guide](docs/PHASE4_USAGE.md) for details.

## Phase 5 Complete ✓

- [x] GPU backend abstraction layer
- [x] Vast.ai implementation
- [x] Job packaging system
- [x] Cost controller with budget enforcement
- [x] Results validator

## Phase 6 Complete ✓

- [x] C++ Matrix class with manual memory management
- [x] Activation functions (sigmoid, relu, softmax)
- [x] Backpropagation engine
- [x] SGD with momentum and Adam optimizers
- [x] pybind11 Python bindings
- [x] Performance benchmarks
- [x] Python fallback for systems without C++

## Phase 7 Complete ✓

- [x] User documentation (quickstart, user guide, troubleshooting)
- [x] Error handling with actionable suggestions
- [x] Performance optimization
- [x] Beta testing infrastructure

## Documentation

- [Quick Start](docs/QUICKSTART.md) - Get started in 5 minutes
- [User Guide](docs/USER_GUIDE.md) - Complete documentation
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions
- [Phase 3 Usage](docs/PHASE3_USAGE.md) - Implementation challenges
- [Phase 4 Usage](docs/PHASE4_USAGE.md) - Interactive visualizations
