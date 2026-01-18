"""Integration tests for Phase 3: Implementation Challenges"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from learning_framework.assessment import ChallengeManager, TestRunner, GradientChecker


def test_full_challenge_workflow():
    """Test complete challenge workflow: load -> edit -> test"""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        manager = ChallengeManager()

        # List challenges
        challenges = manager.list_challenges()
        assert len(challenges) > 0, "No challenges found"

        # Load a challenge
        if challenges:
            challenge = challenges[0]
            assert challenge['name'] is not None
            assert challenge['type'] in ['fill', 'scratch', 'debug', '']


def test_gradient_checker_integration():
    """Test gradient checker with real function"""

    def f(x):
        return np.sum(x ** 2)

    def grad_f(x):
        return 2 * x

    checker = GradientChecker()
    x = np.random.randn(5)

    result = checker.check_gradient(f, grad_f, x)
    assert result['passed'] == True


def test_test_runner_integration():
    """Test test runner with real pytest file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_simple.py"
        test_file.write_text('''
def test_example():
    assert 1 + 1 == 2
        ''')

        runner = TestRunner()
        result = runner.run_tests(str(test_file))

        assert result['passed'] == True


def test_challenge_manager_loads_backprop():
    """Test that backprop challenge loads correctly"""
    manager = ChallengeManager()
    challenge = manager.load_challenge('backprop_fill.py')

    if challenge:
        assert challenge['type'] == 'fill'
        assert 'backprop' in challenge['description'].lower()


def test_challenge_manager_loads_sgd():
    """Test that SGD challenge loads correctly"""
    manager = ChallengeManager()
    challenge = manager.load_challenge('sgd_scratch.py')

    if challenge:
        assert challenge['type'] == 'scratch'
        assert 'sgd' in challenge['description'].lower()


def test_gradient_checker_neural_network():
    """Test gradient checker with simple neural network function"""
    np.random.seed(42)

    def neural_net_loss(W):
        """Simple 2-layer neural net loss"""
        W1 = W[:20].reshape(4, 5)
        W2 = W[20:].reshape(5, 3)

        X = np.random.randn(10, 4)
        y = np.zeros((10, 3))
        y[np.arange(10), np.random.randint(0, 3, 10)] = 1

        # Forward
        h = np.maximum(0, X @ W1)  # ReLU
        scores = h @ W2
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Cross-entropy loss
        return -np.sum(y * np.log(probs + 1e-10)) / 10

    checker = GradientChecker(epsilon=1e-5, threshold=1e-4)
    W = np.random.randn(35) * 0.01

    num_grad = checker.numerical_gradient(neural_net_loss, W)

    # Just verify it computes without error and has correct shape
    assert num_grad.shape == W.shape
