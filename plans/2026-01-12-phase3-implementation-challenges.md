# Phase 3: Implementation Challenges Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an automated system for progressive implementation challenges including challenge templates, test runners, numerical gradient checking, and C++ matrix operations.

**Architecture:** Three-tier challenge system (fill-in-blank, from-scratch, debug) with pytest-based automated testing, numerical gradient verification, and initial C++ implementations with pybind11 bindings.

**Tech Stack:** Python 3.10+, pytest, numpy, pybind11, CMake, C++17

---

## Prerequisites

Before starting Phase 3, ensure Phase 1 and Phase 2 are complete:
- ✓ Progress database with concept tracking
- ✓ Configuration management
- ✓ CLI framework with Click + Rich
- ✓ Quiz system for Tier 1 assessment

---

## Task 1: Challenge System Foundation

**Files:**
- Create: `learning_framework/assessment/__init__.py`
- Create: `learning_framework/assessment/challenge.py`
- Create: `learning_framework/assessment/test_runner.py`
- Create: `tests/test_challenge.py`

**Step 1: Write failing test for challenge loading**

Create `tests/test_challenge.py`:

```python
import pytest
import tempfile
from pathlib import Path
from learning_framework.assessment.challenge import ChallengeManager


def test_challenge_manager_loads_challenge():
    """Test challenge manager loads challenge from file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        challenge_dir = Path(tmpdir)

        # Create sample challenge
        challenge_file = challenge_dir / "backprop_fill.py"
        challenge_file.write_text('''
"""Fill-in-the-blank challenge for backpropagation"""

def backprop(x, y, weights):
    """Compute gradients via backpropagation

    Args:
        x: Input data
        y: Target labels
        weights: Dict of weight matrices

    Returns:
        Dict of gradients
    """
    # Forward pass (provided)
    hidden = sigmoid(x @ weights['W1'])
    output = softmax(hidden @ weights['W2'])

    # YOUR CODE: Compute output layer gradient
    grad_output = ___________  # FILL THIS

    # YOUR CODE: Compute hidden layer gradient
    grad_hidden = ___________  # FILL THIS

    return {'W1': grad_W1, 'W2': grad_W2}


def sigmoid(x):
    """Sigmoid activation"""
    import numpy as np
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    """Softmax activation"""
    import numpy as np
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        ''')

        manager = ChallengeManager(challenge_dir)
        challenge = manager.load_challenge('backprop_fill')

        assert challenge is not None
        assert challenge['name'] == 'backprop_fill'
        assert challenge['type'] == 'fill'
        assert 'code' in challenge


def test_challenge_manager_detects_challenge_type():
    """Test challenge manager detects challenge type from filename"""
    with tempfile.TemporaryDirectory() as tmpdir:
        challenge_dir = Path(tmpdir)

        (challenge_dir / "sgd_fill.py").write_text("# Fill challenge")
        (challenge_dir / "sgd_scratch.py").write_text("# From-scratch challenge")
        (challenge_dir / "sgd_debug.py").write_text("# Debug challenge")

        manager = ChallengeManager(challenge_dir)

        assert manager.get_challenge_type('sgd_fill') == 'fill'
        assert manager.get_challenge_type('sgd_scratch') == 'scratch'
        assert manager.get_challenge_type('sgd_debug') == 'debug'


def test_challenge_manager_lists_challenges():
    """Test challenge manager lists available challenges"""
    with tempfile.TemporaryDirectory() as tmpdir:
        challenge_dir = Path(tmpdir)

        (challenge_dir / "backprop_fill.py").write_text("# Challenge 1")
        (challenge_dir / "sgd_scratch.py").write_text("# Challenge 2")
        (challenge_dir / "adam_debug.py").write_text("# Challenge 3")

        manager = ChallengeManager(challenge_dir)
        challenges = manager.list_challenges()

        assert len(challenges) >= 3
        assert 'backprop_fill' in challenges
        assert 'sgd_scratch' in challenges
        assert 'adam_debug' in challenges
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_challenge.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'learning_framework.assessment'"

**Step 3: Create assessment package**

Create `learning_framework/assessment/__init__.py`:

```python
"""Assessment system for implementation challenges"""

from learning_framework.assessment.challenge import ChallengeManager
from learning_framework.assessment.test_runner import TestRunner

__all__ = ['ChallengeManager', 'TestRunner']
```

**Step 4: Implement ChallengeManager**

Create `learning_framework/assessment/challenge.py`:

```python
"""Challenge management for implementation exercises"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Any


class ChallengeManager:
    """Manages challenge templates and metadata"""

    # Challenge type patterns
    CHALLENGE_TYPES = {
        'fill': 'Fill-in-the-blank',
        'scratch': 'From-scratch implementation',
        'debug': 'Debug and fix'
    }

    def __init__(self, challenge_dir: Optional[Path] = None):
        """Initialize challenge manager

        Args:
            challenge_dir: Directory containing challenge files
        """
        if challenge_dir is None:
            challenge_dir = Path.cwd() / 'data' / 'challenges'

        self.challenge_dir = Path(challenge_dir)
        self.challenge_dir.mkdir(parents=True, exist_ok=True)

    def load_challenge(self, name: str) -> Optional[Dict[str, Any]]:
        """Load challenge by name

        Args:
            name: Challenge name (e.g., 'backprop_fill')

        Returns:
            Challenge dictionary or None if not found
        """
        challenge_path = self.challenge_dir / f"{name}.py"

        if not challenge_path.exists():
            return None

        code = challenge_path.read_text(encoding='utf-8')
        challenge_type = self.get_challenge_type(name)

        return {
            'name': name,
            'type': challenge_type,
            'path': str(challenge_path),
            'code': code,
            'description': self._extract_description(code)
        }

    def get_challenge_type(self, name: str) -> str:
        """Determine challenge type from name

        Args:
            name: Challenge name

        Returns:
            Challenge type ('fill', 'scratch', 'debug')
        """
        name_lower = name.lower()

        for type_key in self.CHALLENGE_TYPES.keys():
            if f"_{type_key}" in name_lower or name_lower.endswith(type_key):
                return type_key

        return 'unknown'

    def list_challenges(self) -> List[str]:
        """List all available challenges

        Returns:
            List of challenge names
        """
        if not self.challenge_dir.exists():
            return []

        challenges = []
        for path in self.challenge_dir.glob("*.py"):
            if path.name != '__init__.py':
                challenges.append(path.stem)

        return sorted(challenges)

    def get_challenges_by_type(self, challenge_type: str) -> List[str]:
        """Get challenges of specific type

        Args:
            challenge_type: Type to filter ('fill', 'scratch', 'debug')

        Returns:
            List of challenge names
        """
        all_challenges = self.list_challenges()
        return [
            name for name in all_challenges
            if self.get_challenge_type(name) == challenge_type
        ]

    def _extract_description(self, code: str) -> str:
        """Extract description from docstring

        Args:
            code: Challenge code

        Returns:
            Description text
        """
        # Extract first docstring
        docstring_match = re.search(r'"""(.+?)"""', code, re.DOTALL)
        if docstring_match:
            return docstring_match.group(1).strip()

        return "No description available"

    def copy_to_workspace(self, challenge_name: str, workspace_dir: Path) -> Path:
        """Copy challenge to user workspace for editing

        Args:
            challenge_name: Name of challenge
            workspace_dir: User's workspace directory

        Returns:
            Path to copied file
        """
        challenge = self.load_challenge(challenge_name)
        if challenge is None:
            raise ValueError(f"Challenge not found: {challenge_name}")

        workspace_dir = Path(workspace_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)

        dest_path = workspace_dir / f"{challenge_name}.py"
        dest_path.write_text(challenge['code'], encoding='utf-8')

        return dest_path
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_challenge.py -v
```

Expected: PASS (all tests pass)

**Step 6: Commit**

```bash
git add learning_framework/assessment/ tests/test_challenge.py
git commit -m "feat: add challenge manager for loading implementation exercises"
```

---

## Task 2: Automated Test Runner

**Files:**
- Create: `learning_framework/assessment/test_runner.py`
- Create: `tests/test_test_runner.py`

**Step 1: Write failing test for test runner**

Create `tests/test_test_runner.py`:

```python
import pytest
import tempfile
from pathlib import Path
from learning_framework.assessment.test_runner import TestRunner


def test_test_runner_executes_pytest():
    """Test runner executes pytest on implementation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create simple implementation
        impl_file = workspace / "simple.py"
        impl_file.write_text('''
def add(a, b):
    return a + b
        ''')

        # Create test file
        test_file = workspace / "test_simple.py"
        test_file.write_text('''
from simple import add

def test_add():
    assert add(2, 3) == 5
    assert add(0, 0) == 0
    assert add(-1, 1) == 0
        ''')

        runner = TestRunner()
        result = runner.run_tests(test_file)

        assert result['passed'] == True
        assert result['num_passed'] == 3
        assert result['num_failed'] == 0


def test_test_runner_reports_failures():
    """Test runner reports test failures"""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create buggy implementation
        impl_file = workspace / "buggy.py"
        impl_file.write_text('''
def multiply(a, b):
    return a + b  # BUG: should be a * b
        ''')

        # Create test file
        test_file = workspace / "test_buggy.py"
        test_file.write_text('''
from buggy import multiply

def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(0, 5) == 0
        ''')

        runner = TestRunner()
        result = runner.run_tests(test_file)

        assert result['passed'] == False
        assert result['num_failed'] > 0
        assert 'failures' in result


def test_test_runner_captures_output():
    """Test runner captures stdout/stderr"""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        test_file = workspace / "test_output.py"
        test_file.write_text('''
def test_with_output():
    print("Debug output")
    assert True
        ''')

        runner = TestRunner()
        result = runner.run_tests(test_file)

        assert result['passed'] == True
        assert 'output' in result or 'stdout' in result
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_test_runner.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement TestRunner**

Create `learning_framework/assessment/test_runner.py`:

```python
"""Automated test execution for challenges"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json


class TestRunner:
    """Runs pytest tests and collects results"""

    def __init__(self):
        """Initialize test runner"""
        pass

    def run_tests(
        self,
        test_file: Path,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """Run pytest on test file

        Args:
            test_file: Path to test file
            timeout: Timeout in seconds

        Returns:
            Dictionary with test results
        """
        test_file = Path(test_file)

        if not test_file.exists():
            return {
                'passed': False,
                'error': f"Test file not found: {test_file}"
            }

        # Run pytest with JSON output
        cmd = [
            sys.executable, '-m', 'pytest',
            str(test_file),
            '-v',
            '--tb=short',
            '--json-report',
            '--json-report-file=/tmp/pytest_report.json'
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=test_file.parent,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Parse results
            return self._parse_pytest_output(
                result.returncode,
                result.stdout,
                result.stderr
            )

        except subprocess.TimeoutExpired:
            return {
                'passed': False,
                'error': f'Tests timed out after {timeout}s',
                'timeout': True
            }
        except Exception as e:
            return {
                'passed': False,
                'error': f'Test execution failed: {str(e)}'
            }

    def _parse_pytest_output(
        self,
        return_code: int,
        stdout: str,
        stderr: str
    ) -> Dict[str, Any]:
        """Parse pytest output

        Args:
            return_code: Process return code
            stdout: Standard output
            stderr: Standard error

        Returns:
            Parsed results dictionary
        """
        # Count passed/failed from output
        num_passed = stdout.count(' PASSED')
        num_failed = stdout.count(' FAILED')
        num_errors = stdout.count(' ERROR')

        passed = return_code == 0

        result = {
            'passed': passed,
            'num_passed': num_passed,
            'num_failed': num_failed,
            'num_errors': num_errors,
            'stdout': stdout,
            'stderr': stderr
        }

        # Extract failure details
        if num_failed > 0:
            result['failures'] = self._extract_failures(stdout)

        return result

    def _extract_failures(self, output: str) -> list:
        """Extract failure details from pytest output

        Args:
            output: Pytest output

        Returns:
            List of failure descriptions
        """
        failures = []

        # Simple extraction - look for FAILED lines
        for line in output.split('\n'):
            if 'FAILED' in line:
                failures.append(line.strip())

        return failures

    def run_with_coverage(
        self,
        test_file: Path,
        source_file: Path
    ) -> Dict[str, Any]:
        """Run tests with coverage analysis

        Args:
            test_file: Test file path
            source_file: Source file to measure coverage for

        Returns:
            Results with coverage information
        """
        test_file = Path(test_file)
        source_file = Path(source_file)

        cmd = [
            sys.executable, '-m', 'pytest',
            str(test_file),
            '--cov=' + str(source_file.stem),
            '--cov-report=term'
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=test_file.parent,
                capture_output=True,
                text=True,
                timeout=60
            )

            parsed = self._parse_pytest_output(
                result.returncode,
                result.stdout,
                result.stderr
            )

            # Extract coverage percentage
            parsed['coverage'] = self._extract_coverage(result.stdout)

            return parsed

        except Exception as e:
            return {
                'passed': False,
                'error': f'Coverage test failed: {str(e)}'
            }

    def _extract_coverage(self, output: str) -> Optional[float]:
        """Extract coverage percentage from pytest-cov output

        Args:
            output: Pytest output with coverage

        Returns:
            Coverage percentage or None
        """
        import re

        # Look for "TOTAL    X%"
        match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', output)
        if match:
            return float(match.group(1))

        return None
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_test_runner.py -v
```

Expected: PASS (all tests pass)

**Step 5: Commit**

```bash
git add learning_framework/assessment/test_runner.py tests/test_test_runner.py
git commit -m "feat: add automated test runner with pytest integration"
```

---

## Task 3: Numerical Gradient Checking

**Files:**
- Create: `learning_framework/assessment/gradient_check.py`
- Create: `tests/test_gradient_check.py`

**Step 1: Write failing test for gradient checker**

Create `tests/test_gradient_check.py`:

```python
import pytest
import numpy as np
from learning_framework.assessment.gradient_check import GradientChecker


def test_gradient_checker_simple_function():
    """Test gradient checking on simple quadratic function"""

    def f(x):
        """f(x) = x^2"""
        return np.sum(x ** 2)

    def grad_f(x):
        """Analytical gradient: 2x"""
        return 2 * x

    checker = GradientChecker()
    x = np.array([1.0, 2.0, 3.0])

    result = checker.check_gradient(f, grad_f, x)

    assert result['passed'] == True
    assert result['relative_error'] < 1e-7


def test_gradient_checker_detects_wrong_gradient():
    """Test gradient checker detects incorrect gradients"""

    def f(x):
        """f(x) = x^2"""
        return np.sum(x ** 2)

    def wrong_grad_f(x):
        """WRONG gradient: 3x instead of 2x"""
        return 3 * x

    checker = GradientChecker()
    x = np.array([1.0, 2.0, 3.0])

    result = checker.check_gradient(f, grad_f, x)

    assert result['passed'] == False
    assert result['relative_error'] > 1e-5


def test_gradient_checker_matrix_function():
    """Test gradient checking on matrix function"""

    def f(W):
        """f(W) = ||W||^2"""
        return np.sum(W ** 2)

    def grad_f(W):
        """Gradient: 2W"""
        return 2 * W

    checker = GradientChecker()
    W = np.random.randn(3, 4)

    result = checker.check_gradient(f, grad_f, W)

    assert result['passed'] == True
    assert result['relative_error'] < 1e-7


def test_gradient_checker_computes_numerical_gradient():
    """Test numerical gradient computation"""

    def f(x):
        return np.sum(x ** 3)

    checker = GradientChecker(epsilon=1e-5)
    x = np.array([1.0, 2.0])

    num_grad = checker.numerical_gradient(f, x)
    analytical_grad = 3 * x ** 2

    diff = np.linalg.norm(num_grad - analytical_grad)
    assert diff < 1e-5
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_gradient_check.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement GradientChecker**

Create `learning_framework/assessment/gradient_check.py`:

```python
"""Numerical gradient checking for verifying implementations"""

import numpy as np
from typing import Callable, Dict, Any


class GradientChecker:
    """Numerical gradient verification using finite differences"""

    def __init__(self, epsilon: float = 1e-5, threshold: float = 1e-7):
        """Initialize gradient checker

        Args:
            epsilon: Step size for finite differences
            threshold: Relative error threshold for pass/fail
        """
        self.epsilon = epsilon
        self.threshold = threshold

    def check_gradient(
        self,
        f: Callable,
        grad_f: Callable,
        x: np.ndarray
    ) -> Dict[str, Any]:
        """Check analytical gradient against numerical gradient

        Args:
            f: Function that takes x and returns scalar
            grad_f: Function that computes analytical gradient
            x: Point to check gradient at

        Returns:
            Dictionary with check results
        """
        # Compute analytical gradient
        analytical_grad = grad_f(x)

        # Compute numerical gradient
        numerical_grad = self.numerical_gradient(f, x)

        # Compute relative error
        numerator = np.linalg.norm(analytical_grad - numerical_grad)
        denominator = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad)

        if denominator < 1e-10:
            relative_error = numerator
        else:
            relative_error = numerator / denominator

        passed = relative_error < self.threshold

        return {
            'passed': passed,
            'relative_error': float(relative_error),
            'threshold': self.threshold,
            'analytical_grad': analytical_grad,
            'numerical_grad': numerical_grad,
            'max_diff': float(np.max(np.abs(analytical_grad - numerical_grad)))
        }

    def numerical_gradient(
        self,
        f: Callable,
        x: np.ndarray
    ) -> np.ndarray:
        """Compute numerical gradient using finite differences

        Args:
            f: Function to compute gradient for
            x: Point to compute gradient at

        Returns:
            Numerical gradient array
        """
        grad = np.zeros_like(x, dtype=float)

        # Flatten for iteration
        x_flat = x.flatten()
        grad_flat = grad.flatten()

        for i in range(len(x_flat)):
            # Save original value
            original = x_flat[i]

            # f(x + epsilon)
            x_flat[i] = original + self.epsilon
            f_plus = f(x.reshape(x.shape))

            # f(x - epsilon)
            x_flat[i] = original - self.epsilon
            f_minus = f(x.reshape(x.shape))

            # Central difference
            grad_flat[i] = (f_plus - f_minus) / (2 * self.epsilon)

            # Restore original value
            x_flat[i] = original

        return grad.reshape(x.shape)

    def check_gradient_batch(
        self,
        f: Callable,
        grad_f: Callable,
        x_samples: list
    ) -> Dict[str, Any]:
        """Check gradient at multiple points

        Args:
            f: Function to check
            grad_f: Analytical gradient function
            x_samples: List of points to check

        Returns:
            Aggregated results
        """
        results = []

        for x in x_samples:
            result = self.check_gradient(f, grad_f, x)
            results.append(result)

        all_passed = all(r['passed'] for r in results)
        max_error = max(r['relative_error'] for r in results)
        avg_error = np.mean([r['relative_error'] for r in results])

        return {
            'passed': all_passed,
            'num_checks': len(results),
            'num_passed': sum(1 for r in results if r['passed']),
            'max_error': float(max_error),
            'avg_error': float(avg_error),
            'individual_results': results
        }
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_gradient_check.py -v
```

Expected: PASS (all tests pass)

**Step 5: Commit**

```bash
git add learning_framework/assessment/gradient_check.py tests/test_gradient_check.py
git commit -m "feat: add numerical gradient checking with finite differences"
```

---

## Task 4: Challenge Templates - Fill-in-Blank

**Files:**
- Create: `data/challenges/backprop_fill.py`
- Create: `data/challenges/tests/test_backprop_fill.py`
- Create: `data/baselines/backprop_reference.py`

**Step 1: Create backprop fill-in-blank challenge**

Create `data/challenges/backprop_fill.py`:

```python
"""Fill-in-the-blank: Backpropagation

Complete the missing parts to implement backpropagation for a two-layer network.
"""

import numpy as np


def sigmoid(x):
    """Sigmoid activation function"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x):
    """Derivative of sigmoid"""
    s = sigmoid(x)
    return s * (1 - s)


def softmax(x):
    """Softmax activation (numerically stable)"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    """Cross-entropy loss"""
    batch_size = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-10)) / batch_size


def backprop_fill(X, y, weights):
    """Implement backpropagation for two-layer network

    Args:
        X: Input data, shape (batch_size, input_dim)
        y: One-hot encoded labels, shape (batch_size, num_classes)
        weights: Dict with keys 'W1' (input_dim, hidden_dim) and
                'W2' (hidden_dim, num_classes)

    Returns:
        Dict with gradient for each weight matrix
    """
    batch_size = X.shape[0]

    # Forward pass (PROVIDED)
    z1 = X @ weights['W1']
    a1 = sigmoid(z1)
    z2 = a1 @ weights['W2']
    a2 = softmax(z2)

    # Backward pass - FILL IN THE BLANKS

    # TODO: Compute gradient of loss with respect to z2
    # Hint: For softmax + cross-entropy, dL/dz2 = (a2 - y) / batch_size
    dz2 = ___________  # FILL THIS

    # TODO: Compute gradient for W2
    # Hint: dL/dW2 = a1.T @ dz2
    dW2 = ___________  # FILL THIS

    # TODO: Compute gradient with respect to a1
    # Hint: dL/da1 = dz2 @ W2.T
    da1 = ___________  # FILL THIS

    # TODO: Compute gradient with respect to z1
    # Hint: dL/dz1 = da1 * sigmoid_derivative(z1)
    dz1 = ___________  # FILL THIS

    # TODO: Compute gradient for W1
    # Hint: dL/dW1 = X.T @ dz1
    dW1 = ___________  # FILL THIS

    return {
        'W1': dW1,
        'W2': dW2
    }
```

**Step 2: Create test file for backprop challenge**

Create `data/challenges/tests/test_backprop_fill.py`:

```python
"""Tests for backprop fill-in-blank challenge"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backprop_fill import backprop_fill, sigmoid, softmax, cross_entropy_loss


def test_backprop_gradient_shapes():
    """Test that gradients have correct shapes"""
    np.random.seed(42)

    X = np.random.randn(32, 10)
    y = np.zeros((32, 5))
    y[np.arange(32), np.random.randint(0, 5, 32)] = 1

    weights = {
        'W1': np.random.randn(10, 20) * 0.01,
        'W2': np.random.randn(20, 5) * 0.01
    }

    grads = backprop_fill(X, y, weights)

    assert grads['W1'].shape == weights['W1'].shape
    assert grads['W2'].shape == weights['W2'].shape


def test_backprop_numerical_gradient():
    """Test backprop gradients using numerical gradient checking"""
    from learning_framework.assessment.gradient_check import GradientChecker

    np.random.seed(42)

    X = np.random.randn(16, 5)
    y = np.zeros((16, 3))
    y[np.arange(16), np.random.randint(0, 3, 16)] = 1

    weights = {
        'W1': np.random.randn(5, 10) * 0.01,
        'W2': np.random.randn(10, 3) * 0.01
    }

    def loss_fn(W_flat):
        """Compute loss for flattened weights"""
        W1_size = 5 * 10
        W1 = W_flat[:W1_size].reshape(5, 10)
        W2 = W_flat[W1_size:].reshape(10, 3)

        z1 = X @ W1
        a1 = sigmoid(z1)
        z2 = a1 @ W2
        a2 = softmax(z2)

        return cross_entropy_loss(a2, y)

    def grad_fn(W_flat):
        """Compute gradients using backprop"""
        W1_size = 5 * 10
        W1 = W_flat[:W1_size].reshape(5, 10)
        W2 = W_flat[W1_size:].reshape(10, 3)

        grads = backprop_fill(X, y, {'W1': W1, 'W2': W2})

        return np.concatenate([grads['W1'].flatten(), grads['W2'].flatten()])

    W_flat = np.concatenate([weights['W1'].flatten(), weights['W2'].flatten()])

    checker = GradientChecker(epsilon=1e-5, threshold=1e-5)
    result = checker.check_gradient(loss_fn, grad_fn, W_flat)

    assert result['passed'], f"Gradient check failed with error: {result['relative_error']}"


def test_backprop_convergence():
    """Test that backprop gradients lead to convergence"""
    np.random.seed(42)

    # Simple dataset
    X = np.random.randn(100, 5)
    y = np.zeros((100, 3))
    y[np.arange(100), np.random.randint(0, 3, 100)] = 1

    weights = {
        'W1': np.random.randn(5, 10) * 0.1,
        'W2': np.random.randn(10, 3) * 0.1
    }

    # Train for a few iterations
    learning_rate = 0.1
    losses = []

    for _ in range(50):
        # Forward pass
        z1 = X @ weights['W1']
        a1 = sigmoid(z1)
        z2 = a1 @ weights['W2']
        a2 = softmax(z2)

        loss = cross_entropy_loss(a2, y)
        losses.append(loss)

        # Backward pass
        grads = backprop_fill(X, y, weights)

        # Update weights
        weights['W1'] -= learning_rate * grads['W1']
        weights['W2'] -= learning_rate * grads['W2']

    # Loss should decrease
    assert losses[-1] < losses[0], "Loss did not decrease during training"
    assert losses[-1] < 1.0, f"Final loss too high: {losses[-1]}"
```

**Step 3: Create reference implementation**

Create `data/baselines/backprop_reference.py`:

```python
"""Reference implementation for backpropagation"""

import numpy as np


def sigmoid(x):
    """Sigmoid activation function"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x):
    """Derivative of sigmoid"""
    s = sigmoid(x)
    return s * (1 - s)


def softmax(x):
    """Softmax activation"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    """Cross-entropy loss"""
    batch_size = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-10)) / batch_size


def backprop_reference(X, y, weights):
    """Reference backpropagation implementation

    Args:
        X: Input data, shape (batch_size, input_dim)
        y: One-hot labels, shape (batch_size, num_classes)
        weights: Dict with 'W1', 'W2'

    Returns:
        Dict with gradients
    """
    batch_size = X.shape[0]

    # Forward pass
    z1 = X @ weights['W1']
    a1 = sigmoid(z1)
    z2 = a1 @ weights['W2']
    a2 = softmax(z2)

    # Backward pass
    dz2 = (a2 - y) / batch_size
    dW2 = a1.T @ dz2

    da1 = dz2 @ weights['W2'].T
    dz1 = da1 * sigmoid_derivative(z1)
    dW1 = X.T @ dz1

    return {
        'W1': dW1,
        'W2': dW2
    }
```

**Step 4: Test the challenge and tests**

```bash
# This should fail initially (blanks not filled)
# pytest data/challenges/tests/test_backprop_fill.py -v
```

**Step 5: Create solution file (for teacher reference)**

Create `data/challenges/solutions/backprop_fill_solution.py`:

```python
"""SOLUTION: Fill-in-the-blank backpropagation"""

# ... (copy full file from reference with blanks filled in)

def backprop_fill(X, y, weights):
    batch_size = X.shape[0]

    # Forward pass
    z1 = X @ weights['W1']
    a1 = sigmoid(z1)
    z2 = a1 @ weights['W2']
    a2 = softmax(z2)

    # Backward pass - SOLUTION
    dz2 = (a2 - y) / batch_size
    dW2 = a1.T @ dz2
    da1 = dz2 @ weights['W2'].T
    dz1 = da1 * sigmoid_derivative(z1)
    dW1 = X.T @ dz1

    return {'W1': dW1, 'W2': dW2}
```

**Step 6: Commit**

```bash
git add data/challenges/ data/baselines/
git commit -m "feat: add backprop fill-in-blank challenge with tests"
```

---

## Task 5: Challenge Templates - From-Scratch

**Files:**
- Create: `data/challenges/sgd_scratch.py`
- Create: `data/challenges/tests/test_sgd_scratch.py`
- Create: `data/baselines/sgd_reference.py`

**Step 1: Create SGD from-scratch challenge**

Create `data/challenges/sgd_scratch.py`:

```python
"""From-scratch: SGD Optimizer with Momentum

Implement SGD optimizer with momentum from scratch.

Requirements:
- Initialize velocity with zeros
- Update rule: velocity = momentum * velocity - lr * gradient
- Parameter update: param = param + velocity
- Support for dictionary of parameters
"""

import numpy as np


class SGDMomentum:
    """SGD optimizer with momentum

    TODO: Implement this class from scratch
    """

    def __init__(self, learning_rate=0.01, momentum=0.9):
        """Initialize optimizer

        Args:
            learning_rate: Learning rate (default: 0.01)
            momentum: Momentum coefficient (default: 0.9)
        """
        # TODO: Store hyperparameters
        # TODO: Initialize velocity dictionary
        pass

    def step(self, params, grads):
        """Perform one optimization step

        Args:
            params: Dictionary of parameters {name: array}
            grads: Dictionary of gradients {name: array}

        Returns:
            Updated parameters dictionary
        """
        # TODO: For each parameter:
        #   1. Initialize velocity if first time
        #   2. Update velocity: v = momentum * v - lr * grad
        #   3. Update parameter: param = param + v
        # TODO: Return updated parameters
        pass

    def zero_grad(self):
        """Reset velocities (optional)"""
        # TODO: Clear velocity dictionary
        pass
```

**Step 2: Create test file for SGD challenge**

Create `data/challenges/tests/test_sgd_scratch.py`:

```python
"""Tests for SGD momentum from-scratch challenge"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sgd_scratch import SGDMomentum


def test_sgd_initialization():
    """Test optimizer initializes correctly"""
    optimizer = SGDMomentum(learning_rate=0.01, momentum=0.9)

    assert hasattr(optimizer, 'learning_rate') or hasattr(optimizer, 'lr')
    assert hasattr(optimizer, 'momentum')


def test_sgd_simple_quadratic():
    """Test SGD on simple quadratic function"""
    np.random.seed(42)

    # Minimize f(x) = ||x||^2
    params = {'x': np.array([5.0, -3.0, 2.0])}

    optimizer = SGDMomentum(learning_rate=0.1, momentum=0.9)

    for _ in range(100):
        # Gradient of ||x||^2 is 2x
        grads = {'x': 2 * params['x']}
        params = optimizer.step(params, grads)

    # Should converge close to zero
    assert np.linalg.norm(params['x']) < 0.1


def test_sgd_momentum_acceleration():
    """Test that momentum accelerates convergence vs vanilla SGD"""
    np.random.seed(42)

    # Start point
    x_start = np.array([10.0, -8.0])

    # SGD with momentum
    params_mom = {'x': x_start.copy()}
    opt_mom = SGDMomentum(learning_rate=0.01, momentum=0.9)

    losses_mom = []
    for _ in range(50):
        loss = np.sum(params_mom['x'] ** 2)
        losses_mom.append(loss)
        grads = {'x': 2 * params_mom['x']}
        params_mom = opt_mom.step(params_mom, grads)

    # SGD without momentum
    params_vanilla = {'x': x_start.copy()}
    opt_vanilla = SGDMomentum(learning_rate=0.01, momentum=0.0)

    losses_vanilla = []
    for _ in range(50):
        loss = np.sum(params_vanilla['x'] ** 2)
        losses_vanilla.append(loss)
        grads = {'x': 2 * params_vanilla['x']}
        params_vanilla = opt_vanilla.step(params_vanilla, grads)

    # Momentum should converge faster (lower final loss)
    assert losses_mom[-1] < losses_vanilla[-1]


def test_sgd_multiple_parameters():
    """Test SGD works with multiple parameter arrays"""
    np.random.seed(42)

    params = {
        'W1': np.random.randn(10, 20),
        'W2': np.random.randn(20, 5),
        'b1': np.random.randn(20),
        'b2': np.random.randn(5)
    }

    grads = {k: np.random.randn(*v.shape) * 0.01 for k, v in params.items()}

    optimizer = SGDMomentum(learning_rate=0.01, momentum=0.9)

    # Should handle multiple parameters
    updated = optimizer.step(params, grads)

    assert set(updated.keys()) == set(params.keys())
    for key in params:
        assert updated[key].shape == params[key].shape
        # Parameters should change
        assert not np.allclose(updated[key], params[key])


def test_sgd_numerical_gradient():
    """Test SGD with numerical gradient on simple function"""
    np.random.seed(42)

    # Minimize f(W) = ||W - target||^2
    target = np.array([[1, 2], [3, 4]], dtype=float)
    params = {'W': np.random.randn(2, 2)}

    optimizer = SGDMomentum(learning_rate=0.1, momentum=0.9)

    for _ in range(200):
        # Gradient of ||W - target||^2 is 2(W - target)
        grads = {'W': 2 * (params['W'] - target)}
        params = optimizer.step(params, grads)

    # Should converge close to target
    error = np.linalg.norm(params['W'] - target)
    assert error < 0.01, f"Failed to converge, error: {error}"
```

**Step 3: Create reference implementation**

Create `data/baselines/sgd_reference.py`:

```python
"""Reference implementation for SGD with momentum"""

import numpy as np


class SGDMomentumReference:
    """Reference SGD optimizer with momentum"""

    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def step(self, params, grads):
        """Perform optimization step"""
        updated_params = {}

        for key in params:
            # Initialize velocity if first time
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])

            # Update velocity
            self.velocity[key] = (
                self.momentum * self.velocity[key]
                - self.learning_rate * grads[key]
            )

            # Update parameter
            updated_params[key] = params[key] + self.velocity[key]

        return updated_params

    def zero_grad(self):
        """Reset velocities"""
        self.velocity = {}
```

**Step 4: Create solution file**

Create `data/challenges/solutions/sgd_scratch_solution.py`:

(Copy reference implementation with detailed comments)

**Step 5: Commit**

```bash
git add data/challenges/sgd_scratch.py data/challenges/tests/test_sgd_scratch.py data/baselines/sgd_reference.py
git commit -m "feat: add SGD momentum from-scratch challenge with tests"
```

---

## Task 6: C++ Matrix Operations Foundation

**Files:**
- Create: `cpp/CMakeLists.txt`
- Create: `cpp/include/matrix.h`
- Create: `cpp/src/matrix.cpp`
- Create: `cpp/bindings/pybind11_wrapper.cpp`
- Create: `cpp/tests/test_matrix.cpp`

**Step 1: Create CMake build configuration**

Create `cpp/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.15)
project(learning_core_cpp)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Optimization flags
if(MSVC)
    add_compile_options(/O2 /fp:fast)
else()
    add_compile_options(-O3 -march=native -ffast-math)
endif()

# Source files
set(SOURCES
    src/matrix.cpp
)

# Create Python module
pybind11_add_module(learning_core_cpp ${SOURCES} bindings/pybind11_wrapper.cpp)

# Include directories
target_include_directories(learning_core_cpp PRIVATE include)

# Installation
install(TARGETS learning_core_cpp DESTINATION .)
```

**Step 2: Create Matrix header**

Create `cpp/include/matrix.h`:

```cpp
/**
 * Matrix class with manual memory management
 *
 * Learning goals:
 * - Row-major memory layout
 * - Cache-friendly iteration
 * - Pointer arithmetic
 * - Manual memory management
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>
#include <vector>
#include <stdexcept>

namespace lf {

class Matrix {
public:
    // Constructors
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, float value);
    Matrix(size_t rows, size_t cols, const std::vector<float>& data);

    // Copy constructor and assignment
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);

    // Move constructor and assignment
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;

    // Destructor
    ~Matrix();

    // Element access
    float& at(size_t row, size_t col);
    const float& at(size_t row, size_t col) const;

    float& operator()(size_t row, size_t col);
    const float& operator()(size_t row, size_t col) const;

    // Dimensions
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }

    // Data access
    float* data() { return data_; }
    const float* data() const { return data_; }

    // Matrix operations
    Matrix matmul(const Matrix& other) const;
    Matrix transpose() const;
    Matrix add(const Matrix& other) const;
    Matrix subtract(const Matrix& other) const;
    Matrix multiply(float scalar) const;

    // Element-wise operations
    Matrix elementwise_multiply(const Matrix& other) const;

    // Utility
    void fill(float value);
    void zeros();
    void ones();

    // Conversion
    std::vector<float> to_vector() const;

private:
    size_t rows_;
    size_t cols_;
    float* data_;

    size_t index(size_t row, size_t col) const {
        return row * cols_ + col;
    }
};

} // namespace lf

#endif // MATRIX_H
```

**Step 3: Implement Matrix class**

Create `cpp/src/matrix.cpp`:

```cpp
#include "matrix.h"
#include <cstring>
#include <cmath>
#include <algorithm>

namespace lf {

Matrix::Matrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols) {
    data_ = new float[rows * cols];
}

Matrix::Matrix(size_t rows, size_t cols, float value)
    : rows_(rows), cols_(cols) {
    data_ = new float[rows * cols];
    fill(value);
}

Matrix::Matrix(size_t rows, size_t cols, const std::vector<float>& data)
    : rows_(rows), cols_(cols) {
    if (data.size() != rows * cols) {
        throw std::invalid_argument("Data size mismatch");
    }
    data_ = new float[rows * cols];
    std::copy(data.begin(), data.end(), data_);
}

Matrix::Matrix(const Matrix& other)
    : rows_(other.rows_), cols_(other.cols_) {
    data_ = new float[rows_ * cols_];
    std::copy(other.data_, other.data_ + size(), data_);
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        delete[] data_;

        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = new float[rows_ * cols_];
        std::copy(other.data_, other.data_ + size(), data_);
    }
    return *this;
}

Matrix::Matrix(Matrix&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {
    other.data_ = nullptr;
    other.rows_ = 0;
    other.cols_ = 0;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        delete[] data_;

        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = other.data_;

        other.data_ = nullptr;
        other.rows_ = 0;
        other.cols_ = 0;
    }
    return *this;
}

Matrix::~Matrix() {
    delete[] data_;
}

float& Matrix::at(size_t row, size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data_[index(row, col)];
}

const float& Matrix::at(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data_[index(row, col)];
}

float& Matrix::operator()(size_t row, size_t col) {
    return data_[index(row, col)];
}

const float& Matrix::operator()(size_t row, size_t col) const {
    return data_[index(row, col)];
}

Matrix Matrix::matmul(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    Matrix result(rows_, other.cols_, 0.0f);

    // Simple matrix multiplication (can be optimized)
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < other.cols_; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < cols_; ++k) {
                sum += at(i, k) * other.at(k, j);
            }
            result(i, j) = sum;
        }
    }

    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);

    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result(j, i) = at(i, j);
        }
    }

    return result;
}

Matrix Matrix::add(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    Matrix result(rows_, cols_);

    for (size_t i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }

    return result;
}

Matrix Matrix::subtract(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }

    Matrix result(rows_, cols_);

    for (size_t i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }

    return result;
}

Matrix Matrix::multiply(float scalar) const {
    Matrix result(rows_, cols_);

    for (size_t i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }

    return result;
}

Matrix Matrix::elementwise_multiply(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match");
    }

    Matrix result(rows_, cols_);

    for (size_t i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }

    return result;
}

void Matrix::fill(float value) {
    std::fill(data_, data_ + size(), value);
}

void Matrix::zeros() {
    fill(0.0f);
}

void Matrix::ones() {
    fill(1.0f);
}

std::vector<float> Matrix::to_vector() const {
    return std::vector<float>(data_, data_ + size());
}

} // namespace lf
```

**Step 4: Create pybind11 bindings**

Create `cpp/bindings/pybind11_wrapper.cpp`:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "matrix.h"

namespace py = pybind11;
using namespace lf;

PYBIND11_MODULE(learning_core_cpp, m) {
    m.doc() = "C++ core for learning framework";

    py::class_<Matrix>(m, "Matrix")
        .def(py::init<size_t, size_t>(),
             py::arg("rows"), py::arg("cols"))
        .def(py::init<size_t, size_t, float>(),
             py::arg("rows"), py::arg("cols"), py::arg("value"))
        .def(py::init<size_t, size_t, const std::vector<float>&>(),
             py::arg("rows"), py::arg("cols"), py::arg("data"))

        // Properties
        .def_property_readonly("rows", &Matrix::rows)
        .def_property_readonly("cols", &Matrix::cols)
        .def_property_readonly("size", &Matrix::size)

        // Element access
        .def("at", py::overload_cast<size_t, size_t>(&Matrix::at),
             py::arg("row"), py::arg("col"))
        .def("__call__", py::overload_cast<size_t, size_t>(&Matrix::operator()),
             py::arg("row"), py::arg("col"))

        // Operations
        .def("matmul", &Matrix::matmul)
        .def("transpose", &Matrix::transpose)
        .def("add", &Matrix::add)
        .def("subtract", &Matrix::subtract)
        .def("multiply", &Matrix::multiply)
        .def("elementwise_multiply", &Matrix::elementwise_multiply)

        // Utility
        .def("fill", &Matrix::fill)
        .def("zeros", &Matrix::zeros)
        .def("ones", &Matrix::ones)

        // Numpy conversion
        .def("to_numpy", [](const Matrix& mat) {
            return py::array_t<float>(
                {mat.rows(), mat.cols()},
                {mat.cols() * sizeof(float), sizeof(float)},
                mat.data()
            );
        })
        .def("to_vector", &Matrix::to_vector)

        // String representation
        .def("__repr__", [](const Matrix& mat) {
            return "<Matrix " + std::to_string(mat.rows()) + "x" +
                   std::to_string(mat.cols()) + ">";
        });
}
```

**Step 5: Create build script**

Create `cpp/build.py`:

```python
"""Build C++ extensions"""

import subprocess
import sys
from pathlib import Path


def build_cpp():
    """Build C++ module using CMake"""
    cpp_dir = Path(__file__).parent
    build_dir = cpp_dir / 'build'

    build_dir.mkdir(exist_ok=True)

    print("Configuring CMake...")
    subprocess.run(
        ['cmake', '-B', str(build_dir), '-S', str(cpp_dir)],
        check=True
    )

    print("Building...")
    subprocess.run(
        ['cmake', '--build', str(build_dir), '--config', 'Release'],
        check=True
    )

    print("C++ module built successfully!")


if __name__ == '__main__':
    build_cpp()
```

**Step 6: Test C++ integration**

Create `tests/test_cpp_matrix.py`:

```python
"""Tests for C++ matrix operations"""

import pytest
import numpy as np

# Try to import C++ module
try:
    from learning_core_cpp import Matrix
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ module not built")
def test_matrix_creation():
    """Test matrix creation"""
    mat = Matrix(3, 4)
    assert mat.rows == 3
    assert mat.cols == 4


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ module not built")
def test_matrix_matmul():
    """Test matrix multiplication"""
    A = Matrix(2, 3, [1, 2, 3, 4, 5, 6])
    B = Matrix(3, 2, [7, 8, 9, 10, 11, 12])

    C = A.matmul(B)

    assert C.rows == 2
    assert C.cols == 2

    # Verify result
    C_np = C.to_numpy()
    expected = np.array([[58, 64], [139, 154]])

    np.testing.assert_array_almost_equal(C_np, expected)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ module not built")
def test_matrix_operations():
    """Test basic matrix operations"""
    A = Matrix(2, 2, [1, 2, 3, 4])
    B = Matrix(2, 2, [5, 6, 7, 8])

    # Addition
    C = A.add(B)
    C_np = C.to_numpy()
    np.testing.assert_array_almost_equal(C_np, [[6, 8], [10, 12]])

    # Subtraction
    D = B.subtract(A)
    D_np = D.to_numpy()
    np.testing.assert_array_almost_equal(D_np, [[4, 4], [4, 4]])

    # Scalar multiply
    E = A.multiply(2.0)
    E_np = E.to_numpy()
    np.testing.assert_array_almost_equal(E_np, [[2, 4], [6, 8]])
```

**Step 7: Add C++ build to CLI**

Update `learning_framework/cli.py` to add build command:

```python
@cli.command()
def build():
    """Build C++ extensions"""
    import subprocess
    from pathlib import Path

    cpp_dir = Path.cwd() / 'cpp'

    if not cpp_dir.exists():
        console.print("[red]C++ directory not found[/red]")
        return

    console.print("[cyan]Building C++ extensions...[/cyan]")

    try:
        subprocess.run(
            [sys.executable, str(cpp_dir / 'build.py')],
            check=True
        )
        console.print("[green]✓ C++ build complete![/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Build failed: {e}[/red]")
```

**Step 8: Commit**

```bash
git add cpp/ tests/test_cpp_matrix.py learning_framework/cli.py
git commit -m "feat: add C++ matrix operations with pybind11 bindings"
```

---

## Task 7: CLI Integration for Challenges

**Files:**
- Modify: `learning_framework/cli.py`
- Create: `tests/test_cli_challenge.py`

**Step 1: Add challenge command to CLI**

Update `learning_framework/cli.py`:

```python
# Add imports
from learning_framework.assessment import ChallengeManager, TestRunner
from pathlib import Path
import shutil

# ... existing code ...

@cli.command()
@click.argument('challenge_name', required=False)
@click.option('--type', '-t', type=click.Choice(['fill', 'scratch', 'debug']), help='Filter by challenge type')
@click.option('--list', '-l', 'list_only', is_flag=True, help='List available challenges')
def challenge(challenge_name, type, list_only):
    """Work on implementation challenges

    Examples:
        lf challenge --list             # List all challenges
        lf challenge backprop_fill      # Start backprop challenge
        lf challenge --type scratch     # List from-scratch challenges
    """
    manager = ChallengeManager()

    if list_only or challenge_name is None:
        # List challenges
        if type:
            challenges = manager.get_challenges_by_type(type)
            console.print(f"\n[bold cyan]{type.title()} Challenges:[/bold cyan]\n")
        else:
            challenges = manager.list_challenges()
            console.print("\n[bold cyan]Available Challenges:[/bold cyan]\n")

        if not challenges:
            console.print("[yellow]No challenges found[/yellow]")
            return

        from rich.table import Table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Challenge")
        table.add_column("Type")
        table.add_column("Status")

        for name in challenges:
            ch_type = manager.get_challenge_type(name)
            # TODO: Get status from progress database
            status = "Not started"
            table.add_row(name, ch_type, status)

        console.print(table)
        console.print("\n[dim]Use 'lf challenge <name>' to start[/dim]")
        return

    # Load and start challenge
    challenge_data = manager.load_challenge(challenge_name)

    if challenge_data is None:
        console.print(f"[red]Challenge not found: {challenge_name}[/red]")
        return

    console.print(f"\n[bold cyan]Challenge: {challenge_name}[/bold cyan]")
    console.print(f"[dim]{challenge_data['description']}[/dim]\n")

    # Copy to workspace
    workspace = Path.cwd() / 'user_data' / 'implementations'
    workspace.mkdir(parents=True, exist_ok=True)

    dest_path = manager.copy_to_workspace(challenge_name, workspace)
    console.print(f"[green]✓ Challenge copied to: {dest_path}[/green]")

    # Show next steps
    console.print("\n[bold]Next steps:[/bold]")
    console.print(f"1. Edit: {dest_path}")
    console.print(f"2. Test: lf test {challenge_name}")
    console.print(f"3. Submit: lf submit {challenge_name}")


@cli.command()
@click.argument('challenge_name')
def test(challenge_name):
    """Run tests for a challenge implementation"""

    # Find implementation
    workspace = Path.cwd() / 'user_data' / 'implementations'
    impl_path = workspace / f"{challenge_name}.py"

    if not impl_path.exists():
        console.print(f"[red]Implementation not found: {impl_path}[/red]")
        console.print("[dim]Run 'lf challenge <name>' first[/dim]")
        return

    # Find test file
    test_path = Path.cwd() / 'data' / 'challenges' / 'tests' / f"test_{challenge_name}.py"

    if not test_path.exists():
        console.print(f"[red]Tests not found for: {challenge_name}[/red]")
        return

    console.print(f"[cyan]Running tests for {challenge_name}...[/cyan]\n")

    # Copy implementation to test directory for import
    test_dir = test_path.parent
    shutil.copy(impl_path, test_dir / impl_path.name)

    # Run tests
    runner = TestRunner()
    result = runner.run_tests(test_path)

    # Display results
    if result['passed']:
        console.print(f"\n[bold green]✓ All tests passed! ({result['num_passed']} passed)[/bold green]")
    else:
        console.print(f"\n[bold red]✗ Tests failed ({result['num_failed']} failed, {result['num_passed']} passed)[/bold red]")

        if 'failures' in result:
            console.print("\n[bold]Failures:[/bold]")
            for failure in result['failures']:
                console.print(f"  {failure}")
```

**Step 2: Test CLI integration**

```bash
lf challenge --list
lf challenge backprop_fill
lf test backprop_fill
```

**Step 3: Commit**

```bash
git add learning_framework/cli.py
git commit -m "feat: add CLI commands for challenges and testing"
```

---

## Task 8: Integration Testing & Documentation

**Files:**
- Create: `tests/test_phase3_integration.py`
- Create: `docs/PHASE3_USAGE.md`
- Modify: `README.md`

**Step 1: Write integration tests**

Create `tests/test_phase3_integration.py`:

```python
"""Integration tests for Phase 3: Implementation Challenges"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from learning_framework.assessment import ChallengeManager, TestRunner, GradientChecker


def test_full_challenge_workflow():
    """Test complete challenge workflow: load -> edit -> test"""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        manager = ChallengeManager()

        # List challenges
        challenges = manager.list_challenges()
        assert len(challenges) > 0

        # Load challenge
        if 'backprop_fill' in challenges:
            challenge = manager.load_challenge('backprop_fill')
            assert challenge is not None
            assert challenge['type'] == 'fill'


def test_gradient_checker_integration():
    """Test gradient checker with real function"""

    def f(x):
        return np.sum(x ** 2)

    def grad_f(x):
        return 2 * x

    checker = GradientChecker()
    x = np.random.randn(5)

    result = checker.check_gradient(f, grad_f, x)
    assert result['passed'] == True


def test_test_runner_integration():
    """Test test runner with real pytest file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_simple.py"
        test_file.write_text('''
def test_example():
    assert 1 + 1 == 2
        ''')

        runner = TestRunner()
        result = runner.run_tests(test_file)

        assert result['passed'] == True
```

**Step 2: Create usage documentation**

Create `docs/PHASE3_USAGE.md`:

```markdown
# Phase 3: Implementation Challenges - Usage Guide

## Overview

Phase 3 provides three levels of implementation challenges to test your understanding:

1. **Fill-in-the-blank**: Complete missing parts of provided code
2. **From-scratch**: Implement algorithms from requirements
3. **Debug**: Find and fix bugs in broken implementations

## Getting Started

### List Available Challenges

```bash
lf challenge --list
```

### Start a Challenge

```bash
lf challenge backprop_fill
```

This copies the challenge template to `user_data/implementations/backprop_fill.py`.

### Work on Implementation

Edit the file and fill in the blanks or implement the required functions:

```bash
code user_data/implementations/backprop_fill.py
```

### Test Your Implementation

```bash
lf test backprop_fill
```

This runs automated tests including:
- Unit tests for correctness
- Numerical gradient checking
- Convergence tests
- Performance tests

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

**Example**: `backprop_debug.py`
- Code has intentional bugs
- Tests indicate what's wrong
- Fix all bugs to pass tests

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
4. **Performance tests**: Check runtime requirements

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

## C++ Challenges

### Building C++ Extensions

```bash
lf build
```

This compiles C++ matrix operations using CMake.

### Using C++ Matrix

```python
from learning_core_cpp import Matrix

# Create matrix
A = Matrix(3, 4)
A.fill(1.0)

# Operations
B = Matrix(4, 2, [1, 2, 3, 4, 5, 6, 7, 8])
C = A.matmul(B)

# Convert to numpy
import numpy as np
C_np = C.to_numpy()
```

### C++ Learning Goals

1. **Memory management**: Manual allocation/deallocation
2. **Cache optimization**: Row-major layout
3. **Performance**: Benchmark vs numpy
4. **Safety**: Bounds checking, move semantics

## Progress Tracking

View your progress:

```bash
lf progress
```

Shows:
- Challenges completed
- Test pass rates
- Next recommended challenges

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

### C++ build fails

- Install CMake and pybind11
- Check compiler is installed
- See build logs for details

## Next Steps

After completing Phase 3 challenges:
- Phase 4: Interactive visualizations
- Phase 5: GPU scaling
- Phase 6: Advanced C++ implementations
```

**Step 3: Update main README**

Update `README.md` to add Phase 3 section:

```markdown
## Phase 3 Complete ✓

- [x] Challenge template system (fill, scratch, debug)
- [x] Automated test runner with pytest
- [x] Numerical gradient checking
- [x] Backprop fill-in-blank challenge
- [x] SGD from-scratch challenge
- [x] C++ matrix operations with pybind11

See [Phase 3 Usage Guide](docs/PHASE3_USAGE.md) for details.
```

**Step 4: Run all tests**

```bash
pytest -v
```

Expected: All tests pass

**Step 5: Final commit**

```bash
git add docs/ README.md tests/test_phase3_integration.py
git commit -m "docs: add Phase 3 usage guide and integration tests"
```

---

## Phase 3 Completion Checklist

Run verification:

```bash
# 1. All tests pass
pytest -v

# 2. Challenges work
lf challenge --list
lf challenge backprop_fill
lf test backprop_fill

# 3. Gradient checker works
python -c "from learning_framework.assessment import GradientChecker; print('OK')"

# 4. C++ builds (optional)
lf build
python -c "from learning_core_cpp import Matrix; print('OK')"

# 5. Code quality
black --check learning_framework/ tests/
flake8 learning_framework/ tests/
```

Expected:
- ✓ All Python tests passing
- ✓ Challenge commands work
- ✓ Gradient checker functional
- ✓ C++ builds successfully (or graceful fallback)
- ✓ Code follows style guidelines

---

## Execution Summary

**Plan complete and saved to `docs/plans/2026-01-12-phase3-implementation-challenges.md`.**

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
