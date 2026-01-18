"""Tests for C++ activation functions"""
import pytest
import numpy as np

try:
    from learning_core_cpp import Activations, Matrix
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False


def numpy_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def numpy_softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ module not built")
class TestActivations:
    def test_sigmoid(self):
        data = [0.0, 1.0, -1.0, 5.0, -5.0, 0.5]
        mat = Matrix(2, 3, data)
        result = Activations.sigmoid(mat)
        expected = numpy_sigmoid(np.array(data).reshape(2, 3))
        np.testing.assert_array_almost_equal(result.to_numpy(), expected, decimal=5)

    def test_sigmoid_derivative(self):
        data = [0.0, 1.0, -1.0, 2.0]
        mat = Matrix(2, 2, data)
        result = Activations.sigmoid_derivative(mat)
        sig = numpy_sigmoid(np.array(data).reshape(2, 2))
        expected = sig * (1 - sig)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected, decimal=5)

    def test_relu(self):
        data = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        mat = Matrix(2, 3, data)
        result = Activations.relu(mat)
        expected = np.maximum(0, np.array(data).reshape(2, 3))
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_relu_derivative(self):
        data = [-2.0, 0.0, 2.0, 1.0]
        mat = Matrix(2, 2, data)
        result = Activations.relu_derivative(mat)
        expected = (np.array(data).reshape(2, 2) > 0).astype(float)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_softmax(self):
        data = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        mat = Matrix(2, 3, data)
        result = Activations.softmax(mat)
        expected = numpy_softmax(np.array(data).reshape(2, 3))
        np.testing.assert_array_almost_equal(result.to_numpy(), expected, decimal=5)

    def test_softmax_numerical_stability(self):
        # Large values that would overflow without stability fix
        data = [1000.0, 1001.0, 1002.0]
        mat = Matrix(1, 3, data)
        result = Activations.softmax(mat)
        # Should not be nan/inf
        arr = result.to_numpy()
        assert not np.any(np.isnan(arr))
        assert not np.any(np.isinf(arr))
        # Sum should be 1
        np.testing.assert_almost_equal(np.sum(arr), 1.0)
