# Phase 3: Implementation Challenges - Usage Guide

## Overview

Phase 3 provides three levels of implementation challenges to test your understanding:

1. **Fill-in-the-blank**: Complete missing parts of provided code
2. **From-scratch**: Implement algorithms from requirements
3. **Debug**: Find and fix bugs in broken implementations

## Getting Started

### List Available Challenges

```bash
python -m learning_framework.cli challenge --list
```

Or filter by type:
```bash
python -m learning_framework.cli challenge --type fill
python -m learning_framework.cli challenge --type scratch
```

### Start a Challenge

```bash
python -m learning_framework.cli challenge backprop_fill.py
```

This copies the challenge template to `user_data/implementations/backprop_fill.py`.

### Work on Implementation

Edit the file and fill in the blanks or implement the required functions:

```bash
# Open in your editor
code user_data/implementations/backprop_fill.py
```

### Test Your Implementation

```bash
python -m learning_framework.cli test backprop_fill.py
```

This runs automated tests including:
- Unit tests for correctness
- Numerical gradient checking
- Convergence tests

## Challenge Types

### Fill-in-the-Blank

**Goal**: Complete missing parts of provided code.

**Example**: `backprop_fill.py`
- Forward pass provided
- Implement backward pass gradients
- Tests verify correctness

**Hints**:
- Look at provided code structure
- Use mathematical formulas from docstrings
- Run tests frequently to check progress

### From-Scratch

**Goal**: Implement complete algorithm from requirements.

**Example**: `sgd_scratch.py`
- Requirements specified in docstring
- Implement entire class
- Tests verify correctness and performance

**Hints**:
- Read requirements carefully
- Start with simple implementation
- Optimize after tests pass

### Debug

**Goal**: Find and fix bugs in broken code.

**Hints**:
- Read error messages carefully
- Add print statements to debug
- Check mathematical formulas

## Testing Details

### Automated Tests

All challenges include:

1. **Shape tests**: Verify output dimensions
2. **Numerical gradient tests**: Compare analytical vs numerical gradients
3. **Convergence tests**: Verify algorithm leads to improvement

### Gradient Checking

Numerical gradient checking verifies your analytical gradients:

```python
from learning_framework.assessment import GradientChecker

checker = GradientChecker(epsilon=1e-5, threshold=1e-7)
result = checker.check_gradient(loss_fn, grad_fn, params)

if result['passed']:
    print("Gradients correct!")
else:
    print(f"Error: {result['relative_error']}")
```

## Tips

1. **Start with fill-in-blank**: Easiest level to learn concepts
2. **Use gradient checking**: Catch bugs early
3. **Read test files**: Understand what's being tested
4. **Incremental development**: Make small changes, test frequently
5. **Study reference implementations**: Learn from solutions after passing

## Common Issues

### Tests fail with import errors

Make sure you're in the project root directory.

### Gradient check fails

- Check your mathematical formulas
- Verify array dimensions
- Look for numerical instability (exp overflow, log(0))

## Available Challenges

| Challenge | Type | Description |
|-----------|------|-------------|
| backprop_fill.py | fill | Backpropagation for 2-layer network |
| sgd_scratch.py | scratch | SGD optimizer with momentum |

## Directory Structure

```
data/
├── challenges/          # Challenge templates
│   ├── backprop_fill.py
│   ├── sgd_scratch.py
│   ├── solutions/       # Reference solutions
│   └── tests/           # Test files
├── baselines/           # Reference implementations
user_data/
└── implementations/     # Your working files
```
