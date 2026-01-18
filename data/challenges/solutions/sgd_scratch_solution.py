"""SOLUTION: SGD Optimizer with Momentum

This is the completed solution for the sgd_scratch.py challenge.
"""

import numpy as np


class SGDMomentum:
    """SGD optimizer with momentum

    Momentum helps accelerate convergence by accumulating a velocity vector
    in directions of persistent reduction in the objective.

    The update rule is:
        v_t = momentum * v_{t-1} - learning_rate * gradient
        param = param + v_t
    """

    def __init__(self, learning_rate=0.01, momentum=0.9):
        """Initialize optimizer

        Args:
            learning_rate: Learning rate (default: 0.01)
            momentum: Momentum coefficient (default: 0.9)
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def step(self, params, grads):
        """Perform one optimization step

        Args:
            params: Dictionary of parameters {name: array}
            grads: Dictionary of gradients {name: array}

        Returns:
            Updated parameters dictionary
        """
        updated_params = {}

        for key in params:
            # Initialize velocity if first time seeing this parameter
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])

            # Update velocity: v = momentum * v - lr * grad
            self.velocity[key] = (
                self.momentum * self.velocity[key]
                - self.learning_rate * grads[key]
            )

            # Update parameter: param = param + v
            updated_params[key] = params[key] + self.velocity[key]

        return updated_params

    def zero_grad(self):
        """Reset velocities"""
        self.velocity = {}
