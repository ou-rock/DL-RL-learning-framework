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

    result = checker.check_gradient(f, wrong_grad_f, x)

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
