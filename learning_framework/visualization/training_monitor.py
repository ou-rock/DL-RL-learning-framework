"""Real-time training monitoring and visualization data"""

import time
from typing import Dict, List, Any, Optional
from collections import deque
import numpy as np


class TrainingMonitor:
    """Monitors training progress and generates visualization data"""

    def __init__(self, total_epochs: Optional[int] = None, max_history: int = 1000):
        """Initialize training monitor

        Args:
            total_epochs: Total number of epochs (for time estimation)
            max_history: Maximum history entries to keep
        """
        self.total_epochs = total_epochs
        self.max_history = max_history

        self.history: Dict[str, List[float]] = {
            'epochs': [],
            'loss': [],
            'accuracy': [],
            'lr': [],
            'timestamps': []
        }

        self.start_time = time.time()
        self.epoch_times: deque = deque(maxlen=10)
        self.last_epoch_time = self.start_time

    def record(self, epoch: int, loss: float, accuracy: float = None, lr: float = None, **kwargs):
        """Record training metrics for an epoch

        Args:
            epoch: Current epoch number
            loss: Training loss
            accuracy: Optional accuracy metric
            lr: Optional learning rate
            **kwargs: Additional metrics
        """
        current_time = time.time()

        # Track epoch duration
        epoch_duration = current_time - self.last_epoch_time
        self.epoch_times.append(epoch_duration)
        self.last_epoch_time = current_time

        # Record metrics
        self.history['epochs'].append(epoch)
        self.history['loss'].append(loss)
        self.history['accuracy'].append(accuracy if accuracy is not None else 0)
        self.history['lr'].append(lr if lr is not None else 0)
        self.history['timestamps'].append(current_time)

        # Record additional metrics
        for key, value in kwargs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

        # Trim history if needed
        if len(self.history['epochs']) > self.max_history:
            for key in self.history:
                self.history[key] = self.history[key][-self.max_history:]

    def get_history(self) -> Dict[str, List[float]]:
        """Get training history

        Returns:
            Dictionary of metric histories
        """
        return self.history

    def get_smoothed_loss(self, window: int = 5) -> List[float]:
        """Get smoothed loss using moving average

        Args:
            window: Window size for smoothing

        Returns:
            Smoothed loss values
        """
        losses = self.history['loss']
        if len(losses) < window:
            return losses.copy()

        smoothed = []
        for i in range(len(losses)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(losses[start:i+1]))

        return smoothed

    def estimate_remaining_time(self) -> float:
        """Estimate remaining training time

        Returns:
            Estimated seconds remaining
        """
        if not self.epoch_times or self.total_epochs is None:
            return 0

        current_epoch = len(self.history['epochs'])
        remaining_epochs = self.total_epochs - current_epoch

        if remaining_epochs <= 0:
            return 0

        avg_epoch_time = np.mean(list(self.epoch_times))
        return remaining_epochs * avg_epoch_time

    def detect_plateau(self, threshold: float = 0.001, patience: int = 5) -> bool:
        """Detect if loss has plateaued

        Args:
            threshold: Minimum improvement to not be plateau
            patience: Number of epochs to check

        Returns:
            True if plateau detected
        """
        losses = self.history['loss']

        if len(losses) < patience:
            return False

        recent = losses[-patience:]
        improvement = max(recent) - min(recent)

        return improvement < threshold

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current training statistics

        Returns:
            Current epoch, loss, accuracy, etc.
        """
        if not self.history['epochs']:
            return {}

        stats = {
            'epoch': self.history['epochs'][-1],
            'loss': self.history['loss'][-1],
            'accuracy': self.history['accuracy'][-1],
            'elapsed_time': time.time() - self.start_time,
            'remaining_time': self.estimate_remaining_time(),
            'is_plateau': self.detect_plateau()
        }

        if self.history['lr'][-1]:
            stats['lr'] = self.history['lr'][-1]

        return stats

    def to_json(self) -> Dict[str, Any]:
        """Serialize for API/WebSocket response

        Returns:
            JSON-serializable dictionary
        """
        return {
            'history': {
                'epochs': self.history['epochs'],
                'loss': self.history['loss'],
                'accuracy': self.history['accuracy'],
                'lr': self.history['lr']
            },
            'smoothed_loss': self.get_smoothed_loss(),
            'current': self.get_current_stats(),
            'total_epochs': self.total_epochs
        }

    def get_update_message(self) -> Dict[str, Any]:
        """Get WebSocket update message

        Returns:
            Message for real-time updates
        """
        return {
            'type': 'training_update',
            'payload': {
                'epoch': self.history['epochs'][-1] if self.history['epochs'] else 0,
                'loss': self.history['loss'][-1] if self.history['loss'] else 0,
                'accuracy': self.history['accuracy'][-1] if self.history['accuracy'] else 0,
                'elapsed': time.time() - self.start_time,
                'remaining': self.estimate_remaining_time()
            }
        }
