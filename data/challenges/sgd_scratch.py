"""From-scratch: SGD Optimizer with Momentum

Implement SGD optimizer with momentum from scratch.

Requirements:
- Initialize velocity with zeros
- Update rule: velocity = momentum * velocity - lr * gradient
- Parameter update: param = param + velocity
- Support for dictionary of parameters
"""

import numpy as np


class SGDMomentum:
    """SGD optimizer with momentum

    TODO: Implement this class from scratch
    """

    def __init__(self, learning_rate=0.01, momentum=0.9):
        """Initialize optimizer

        Args:
            learning_rate: Learning rate (default: 0.01)
            momentum: Momentum coefficient (default: 0.9)
        """
        # TODO: Store hyperparameters
        # TODO: Initialize velocity dictionary
        pass

    def step(self, params, grads):
        """Perform one optimization step

        Args:
            params: Dictionary of parameters {name: array}
            grads: Dictionary of gradients {name: array}

        Returns:
            Updated parameters dictionary
        """
        # TODO: For each parameter:
        #   1. Initialize velocity if first time
        #   2. Update velocity: v = momentum * v - lr * grad
        #   3. Update parameter: param = param + v
        # TODO: Return updated parameters
        pass

    def zero_grad(self):
        """Reset velocities (optional)"""
        # TODO: Clear velocity dictionary
        pass
