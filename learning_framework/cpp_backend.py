"""C++ backend with Python fallback"""

import numpy as np

# Try to import C++ module
try:
    from learning_core_cpp import Matrix, Activations, BackpropEngine, SGDMomentum, Adam
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False


def get_backend_status():
    """Return backend availability status"""
    return {
        'cpp_available': CPP_AVAILABLE,
        'backend': 'C++' if CPP_AVAILABLE else 'Python'
    }


# Python fallbacks if C++ not available
if not CPP_AVAILABLE:
    class Matrix:
        """Python fallback for Matrix"""
        def __init__(self, rows, cols, data=None):
            self._rows = rows
            self._cols = cols
            if data is None:
                self._data = np.zeros((rows, cols), dtype=np.float32)
            elif isinstance(data, (int, float)):
                self._data = np.full((rows, cols), data, dtype=np.float32)
            else:
                self._data = np.array(data, dtype=np.float32).reshape(rows, cols)

        @property
        def rows(self): return self._rows

        @property
        def cols(self): return self._cols

        def to_numpy(self): return self._data.copy()

        def to_vector(self): return self._data.flatten().tolist()

        def matmul(self, other):
            result = Matrix(self._rows, other._cols)
            result._data = self._data @ other._data
            return result

        def transpose(self):
            result = Matrix(self._cols, self._rows)
            result._data = self._data.T.copy()
            return result

        def add(self, other):
            result = Matrix(self._rows, self._cols)
            result._data = self._data + other._data
            return result

        def subtract(self, other):
            result = Matrix(self._rows, self._cols)
            result._data = self._data - other._data
            return result

        def multiply(self, scalar):
            result = Matrix(self._rows, self._cols)
            result._data = self._data * scalar
            return result

        def elementwise_multiply(self, other):
            result = Matrix(self._rows, self._cols)
            result._data = self._data * other._data
            return result


    class Activations:
        """Python fallback for Activations"""
        @staticmethod
        def sigmoid(x):
            result = Matrix(x.rows, x.cols)
            result._data = 1.0 / (1.0 + np.exp(-np.clip(x._data, -500, 500)))
            return result

        @staticmethod
        def sigmoid_derivative(x):
            sig = Activations.sigmoid(x)
            result = Matrix(x.rows, x.cols)
            result._data = sig._data * (1 - sig._data)
            return result

        @staticmethod
        def relu(x):
            result = Matrix(x.rows, x.cols)
            result._data = np.maximum(0, x._data)
            return result

        @staticmethod
        def relu_derivative(x):
            result = Matrix(x.rows, x.cols)
            result._data = (x._data > 0).astype(np.float32)
            return result

        @staticmethod
        def softmax(x):
            result = Matrix(x.rows, x.cols)
            exp_x = np.exp(x._data - np.max(x._data, axis=-1, keepdims=True))
            result._data = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
            return result
