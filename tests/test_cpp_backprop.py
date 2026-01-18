"""Tests for C++ backpropagation engine"""
import pytest
import numpy as np

try:
    from learning_core_cpp import BackpropEngine, Matrix
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ module not built")
class TestBackpropEngine:
    def test_two_layer_forward(self):
        np.random.seed(42)
        X = np.random.randn(4, 3).astype(np.float32)
        W1 = np.random.randn(3, 5).astype(np.float32) * 0.1
        W2 = np.random.randn(5, 2).astype(np.float32) * 0.1

        engine = BackpropEngine()
        engine.set_weights('W1', Matrix(3, 5, W1.flatten().tolist()))
        engine.set_weights('W2', Matrix(5, 2, W2.flatten().tolist()))

        X_mat = Matrix(4, 3, X.flatten().tolist())
        output = engine.forward(X_mat)

        # Output should be (4, 2) with softmax applied
        assert output.rows == 4
        assert output.cols == 2
        # Each row should sum to ~1 (softmax)
        arr = output.to_numpy()
        np.testing.assert_array_almost_equal(np.sum(arr, axis=1), [1, 1, 1, 1], decimal=5)

    def test_backward_gradient_shapes(self):
        np.random.seed(42)
        X = np.random.randn(4, 3).astype(np.float32)
        y = np.zeros((4, 2), dtype=np.float32)
        y[np.arange(4), np.random.randint(0, 2, 4)] = 1

        W1 = np.random.randn(3, 5).astype(np.float32) * 0.1
        W2 = np.random.randn(5, 2).astype(np.float32) * 0.1

        engine = BackpropEngine()
        engine.set_weights('W1', Matrix(3, 5, W1.flatten().tolist()))
        engine.set_weights('W2', Matrix(5, 2, W2.flatten().tolist()))

        X_mat = Matrix(4, 3, X.flatten().tolist())
        y_mat = Matrix(4, 2, y.flatten().tolist())

        engine.forward(X_mat)
        grads = engine.backward(y_mat)

        assert 'W1' in grads
        assert 'W2' in grads
        assert grads['W1'].rows == 3
        assert grads['W1'].cols == 5
        assert grads['W2'].rows == 5
        assert grads['W2'].cols == 2

    def test_backward_numerical_gradient(self):
        """Verify analytical gradients match numerical gradients"""
        np.random.seed(42)
        X = np.random.randn(8, 3).astype(np.float32)
        y = np.zeros((8, 2), dtype=np.float32)
        y[np.arange(8), np.random.randint(0, 2, 8)] = 1

        W1 = np.random.randn(3, 5).astype(np.float32) * 0.1
        W2 = np.random.randn(5, 2).astype(np.float32) * 0.1

        engine = BackpropEngine()
        engine.set_weights('W1', Matrix(3, 5, W1.flatten().tolist()))
        engine.set_weights('W2', Matrix(5, 2, W2.flatten().tolist()))

        X_mat = Matrix(8, 3, X.flatten().tolist())
        y_mat = Matrix(8, 2, y.flatten().tolist())

        # Analytical gradient
        engine.forward(X_mat)
        grads = engine.backward(y_mat)
        dW1_analytical = np.array(grads['W1'].to_vector()).reshape(3, 5)

        # Numerical gradient for W1
        epsilon = 1e-5
        dW1_numerical = np.zeros_like(W1)

        for i in range(W1.shape[0]):
            for j in range(W1.shape[1]):
                W1_plus = W1.copy()
                W1_plus[i, j] += epsilon
                W1_minus = W1.copy()
                W1_minus[i, j] -= epsilon

                engine.set_weights('W1', Matrix(3, 5, W1_plus.flatten().tolist()))
                out_plus = engine.forward(X_mat)
                loss_plus = engine.cross_entropy_loss(out_plus, y_mat)

                engine.set_weights('W1', Matrix(3, 5, W1_minus.flatten().tolist()))
                out_minus = engine.forward(X_mat)
                loss_minus = engine.cross_entropy_loss(out_minus, y_mat)

                dW1_numerical[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

        # Compare
        rel_error = np.linalg.norm(dW1_analytical - dW1_numerical) / (
            np.linalg.norm(dW1_analytical) + np.linalg.norm(dW1_numerical) + 1e-8
        )
        assert rel_error < 1e-5, f"Gradient check failed with relative error: {rel_error}"
