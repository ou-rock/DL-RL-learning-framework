"""Tests for C++ Matrix class"""
import pytest
import numpy as np

try:
    from learning_core_cpp import Matrix
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ module not built")
class TestMatrix:
    def test_creation(self):
        mat = Matrix(3, 4)
        assert mat.rows == 3
        assert mat.cols == 4

    def test_creation_with_value(self):
        mat = Matrix(2, 2, 5.0)
        arr = mat.to_numpy()
        np.testing.assert_array_equal(arr, [[5, 5], [5, 5]])

    def test_matmul_correct(self):
        A = Matrix(2, 3, [1, 2, 3, 4, 5, 6])
        B = Matrix(3, 2, [7, 8, 9, 10, 11, 12])
        C = A.matmul(B)
        expected = np.array([[58, 64], [139, 154]], dtype=np.float32)
        np.testing.assert_array_almost_equal(C.to_numpy(), expected)

    def test_transpose(self):
        A = Matrix(2, 3, [1, 2, 3, 4, 5, 6])
        AT = A.transpose()
        assert AT.rows == 3
        assert AT.cols == 2

    def test_elementwise_ops(self):
        A = Matrix(2, 2, [1, 2, 3, 4])
        B = Matrix(2, 2, [5, 6, 7, 8])

        # Add
        C = A.add(B)
        np.testing.assert_array_almost_equal(C.to_numpy(), [[6, 8], [10, 12]])

        # Scalar multiply
        D = A.multiply(2.0)
        np.testing.assert_array_almost_equal(D.to_numpy(), [[2, 4], [6, 8]])
