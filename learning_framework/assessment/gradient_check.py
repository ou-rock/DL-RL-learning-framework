"""Numerical gradient checking for verifying implementations"""

import numpy as np
from typing import Callable, Dict, Any


class GradientChecker:
    """Numerical gradient verification using finite differences"""

    def __init__(self, epsilon: float = 1e-5, threshold: float = 1e-7):
        """Initialize gradient checker

        Args:
            epsilon: Step size for finite differences
            threshold: Relative error threshold for pass/fail
        """
        self.epsilon = epsilon
        self.threshold = threshold

    def check_gradient(
        self,
        f: Callable,
        grad_f: Callable,
        x: np.ndarray
    ) -> Dict[str, Any]:
        """Check analytical gradient against numerical gradient

        Args:
            f: Function that takes x and returns scalar
            grad_f: Function that computes analytical gradient
            x: Point to check gradient at

        Returns:
            Dictionary with check results
        """
        # Compute analytical gradient
        analytical_grad = grad_f(x)

        # Compute numerical gradient
        numerical_grad = self.numerical_gradient(f, x)

        # Compute relative error
        numerator = np.linalg.norm(analytical_grad - numerical_grad)
        denominator = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad)

        if denominator < 1e-10:
            relative_error = numerator
        else:
            relative_error = numerator / denominator

        passed = relative_error < self.threshold

        return {
            'passed': passed,
            'relative_error': float(relative_error),
            'threshold': self.threshold,
            'analytical_grad': analytical_grad,
            'numerical_grad': numerical_grad,
            'max_diff': float(np.max(np.abs(analytical_grad - numerical_grad)))
        }

    def numerical_gradient(
        self,
        f: Callable,
        x: np.ndarray
    ) -> np.ndarray:
        """Compute numerical gradient using finite differences

        Args:
            f: Function to compute gradient for
            x: Point to compute gradient at

        Returns:
            Numerical gradient array
        """
        grad = np.zeros_like(x, dtype=float)

        # Work with a copy to avoid modifying input
        x_perturbed = x.copy()

        # Flatten for iteration
        it = np.nditer(x, flags=['multi_index'])

        while not it.finished:
            idx = it.multi_index
            original = x_perturbed[idx]

            # f(x + epsilon)
            x_perturbed[idx] = original + self.epsilon
            f_plus = f(x_perturbed)

            # f(x - epsilon)
            x_perturbed[idx] = original - self.epsilon
            f_minus = f(x_perturbed)

            # Central difference
            grad[idx] = (f_plus - f_minus) / (2 * self.epsilon)

            # Restore original value
            x_perturbed[idx] = original

            it.iternext()

        return grad

    def check_gradient_batch(
        self,
        f: Callable,
        grad_f: Callable,
        x_samples: list
    ) -> Dict[str, Any]:
        """Check gradient at multiple points

        Args:
            f: Function to check
            grad_f: Analytical gradient function
            x_samples: List of points to check

        Returns:
            Aggregated results
        """
        results = []

        for x in x_samples:
            result = self.check_gradient(f, grad_f, x)
            results.append(result)

        all_passed = all(r['passed'] for r in results)
        max_error = max(r['relative_error'] for r in results)
        avg_error = np.mean([r['relative_error'] for r in results])

        return {
            'passed': all_passed,
            'num_checks': len(results),
            'num_passed': sum(1 for r in results if r['passed']),
            'max_error': float(max_error),
            'avg_error': float(avg_error),
            'individual_results': results
        }
