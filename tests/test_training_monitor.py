import pytest
import numpy as np
import time
from learning_framework.visualization.training_monitor import TrainingMonitor


def test_training_monitor_records_metrics():
    """Test training monitor records metrics"""
    monitor = TrainingMonitor()

    monitor.record(epoch=1, loss=0.5, accuracy=0.6)
    monitor.record(epoch=2, loss=0.3, accuracy=0.75)

    history = monitor.get_history()

    assert len(history['epochs']) == 2
    assert history['loss'][0] == 0.5
    assert history['accuracy'][1] == 0.75


def test_training_monitor_computes_smoothed():
    """Test training monitor computes smoothed values"""
    monitor = TrainingMonitor()

    for i in range(10):
        monitor.record(epoch=i, loss=1.0 / (i + 1))

    smoothed = monitor.get_smoothed_loss(window=3)

    assert len(smoothed) == 10
    # Smoothed values should be present
    assert smoothed[-1] > 0


def test_training_monitor_estimates_remaining():
    """Test training monitor estimates remaining time"""
    monitor = TrainingMonitor(total_epochs=10)

    # Simulate epochs with timing
    for i in range(3):
        monitor.record(epoch=i, loss=0.5)
        time.sleep(0.01)  # Simulate training time

    remaining = monitor.estimate_remaining_time()

    assert remaining > 0
    assert remaining < 100  # Should be reasonable


def test_training_monitor_detects_plateau():
    """Test training monitor detects loss plateau"""
    monitor = TrainingMonitor()

    # Simulate plateau
    for i in range(10):
        monitor.record(epoch=i, loss=0.5 + np.random.normal(0, 0.001))

    is_plateau = monitor.detect_plateau(threshold=0.01, patience=5)
    assert is_plateau == True

    # Clear and simulate improvement
    monitor = TrainingMonitor()
    for i in range(10):
        monitor.record(epoch=i, loss=1.0 - i * 0.1)

    is_plateau = monitor.detect_plateau(threshold=0.01, patience=5)
    assert is_plateau == False


def test_training_monitor_json_output():
    """Test training monitor produces JSON for visualization"""
    monitor = TrainingMonitor()

    monitor.record(epoch=1, loss=0.5, accuracy=0.6, lr=0.01)
    monitor.record(epoch=2, loss=0.3, accuracy=0.75, lr=0.01)

    json_data = monitor.to_json()

    assert 'history' in json_data
    assert 'current' in json_data
    assert json_data['current']['epoch'] == 2
