"""Tests for C++ optimizers"""
import pytest
import numpy as np

try:
    from learning_core_cpp import SGDMomentum, Adam, Matrix
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ module not built")
class TestSGDMomentum:
    def test_creation(self):
        opt = SGDMomentum(learning_rate=0.01, momentum=0.9)
        assert opt.learning_rate == 0.01
        assert opt.momentum == 0.9

    def test_simple_convergence(self):
        """Test SGD converges on simple quadratic"""
        opt = SGDMomentum(learning_rate=0.1, momentum=0.9)

        # Minimize ||x||^2
        x = Matrix(1, 3, [5.0, -3.0, 2.0])

        for _ in range(100):
            # Gradient is 2x
            grad = x.multiply(2.0)
            x = opt.step('x', x, grad)

        arr = x.to_numpy().flatten()
        assert np.linalg.norm(arr) < 0.1

    def test_momentum_acceleration(self):
        """Test momentum accelerates convergence"""
        x_start = [10.0, -8.0, 5.0]

        # With momentum
        opt_mom = SGDMomentum(learning_rate=0.01, momentum=0.9)
        x_mom = Matrix(1, 3, x_start)
        losses_mom = []
        for _ in range(50):
            loss = sum(v**2 for v in x_mom.to_vector())
            losses_mom.append(loss)
            grad = x_mom.multiply(2.0)
            x_mom = opt_mom.step('x', x_mom, grad)

        # Without momentum
        opt_no = SGDMomentum(learning_rate=0.01, momentum=0.0)
        x_no = Matrix(1, 3, x_start)
        losses_no = []
        for _ in range(50):
            loss = sum(v**2 for v in x_no.to_vector())
            losses_no.append(loss)
            grad = x_no.multiply(2.0)
            x_no = opt_no.step('x', x_no, grad)

        # Momentum should converge faster
        assert losses_mom[-1] < losses_no[-1]


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ module not built")
class TestAdam:
    def test_creation(self):
        opt = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
        assert opt.learning_rate == 0.001

    def test_simple_convergence(self):
        """Test Adam converges on simple quadratic"""
        opt = Adam(learning_rate=0.1)

        x = Matrix(1, 3, [5.0, -3.0, 2.0])

        for _ in range(200):
            grad = x.multiply(2.0)
            x = opt.step('x', x, grad)

        arr = x.to_numpy().flatten()
        assert np.linalg.norm(arr) < 0.1

    def test_sparse_gradient_handling(self):
        """Test Adam handles sparse gradients well"""
        opt = Adam(learning_rate=0.1)

        x = Matrix(1, 4, [1.0, 2.0, 3.0, 4.0])

        for i in range(100):
            # Sparse gradient - only update some dimensions
            grad_data = [0.0, 0.0, 0.0, 0.0]
            grad_data[i % 4] = 2.0 * x.to_vector()[i % 4]
            grad = Matrix(1, 4, grad_data)
            x = opt.step('x', x, grad)

        # Should still converge
        arr = x.to_numpy().flatten()
        assert np.linalg.norm(arr) < 1.0
