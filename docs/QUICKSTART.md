# Quick Start Guide

Get started with the Learning Framework in 5 minutes.

## Prerequisites

- Python 3.10+
- pip

## Installation

```bash
# Clone repository
git clone <repository-url>
cd learning-framework

# Install in development mode
pip install -e .

# Verify installation
lf --version
```

## First Steps

### 1. Set Up Configuration

```bash
# Copy example config
cp user_data/config-example.yaml user_data/config.yaml

# Edit to add your materials directory
# (Use your preferred editor)
```

### 2. Index Your Materials

```bash
lf index
```

This scans your configured directories for learning materials.

### 3. Start Learning

```bash
lf learn
```

Select a concept from the menu and begin your learning journey!

### 4. Take a Quiz

```bash
# Quiz on a specific concept
lf quiz --concept backpropagation

# Daily review (spaced repetition)
lf quiz
```

### 5. Try a Challenge

```bash
# List available challenges
lf challenge --list

# Start a challenge
lf challenge backprop_fill
```

## What's Next?

- Read the [User Guide](USER_GUIDE.md) for detailed documentation
- Check [Troubleshooting](TROUBLESHOOTING.md) if you encounter issues
- Explore [Phase 3 Usage](PHASE3_USAGE.md) for implementation challenges
- Learn about [Visualizations](PHASE4_USAGE.md) for interactive learning

## Common Commands

| Command | Description |
|---------|-------------|
| `lf learn` | Interactive learning session |
| `lf quiz` | Take quiz / daily review |
| `lf challenge --list` | List implementation challenges |
| `lf challenge <name>` | Start a challenge |
| `lf test <name>` | Test your implementation |
| `lf viz <concept>` | Launch visualization server |
| `lf progress` | View learning progress |
| `lf config` | View/edit configuration |
| `lf index` | Re-index learning materials |

## Getting Help

```bash
# Show all commands
lf --help

# Help for specific command
lf learn --help
```

Need more help? Check the [Troubleshooting Guide](TROUBLESHOOTING.md).
