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
