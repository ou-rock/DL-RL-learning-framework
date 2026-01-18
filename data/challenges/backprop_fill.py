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
