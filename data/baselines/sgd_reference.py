"""Reference implementation for SGD with momentum"""

import numpy as np


class SGDMomentumReference:
    """Reference SGD optimizer with momentum"""

    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def step(self, params, grads):
        """Perform optimization step"""
        updated_params = {}

        for key in params:
            # Initialize velocity if first time
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])

            # Update velocity
            self.velocity[key] = (
                self.momentum * self.velocity[key]
                - self.learning_rate * grads[key]
            )

            # Update parameter
            updated_params[key] = params[key] + self.velocity[key]

        return updated_params

    def zero_grad(self):
        """Reset velocities"""
        self.velocity = {}
