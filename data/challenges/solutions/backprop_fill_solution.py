"""SOLUTION: Fill-in-the-blank backpropagation

This is the completed solution for the backprop_fill.py challenge.
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
