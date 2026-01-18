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
