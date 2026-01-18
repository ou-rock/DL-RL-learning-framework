"""Abstract base class for GPU backends"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class JobConfig:
    """Configuration for a GPU job"""
    concept: str
    implementation_path: str
    dataset: str
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 0.001
    target_accuracy: float = 0.90
    max_runtime_minutes: int = 60
    requirements: List[str] = field(default_factory=lambda: ['numpy', 'torch'])
    extra_files: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobStatus:
    """Status of a running or completed job"""
    job_id: str
    status: str  # 'pending', 'starting', 'running', 'completed', 'failed', 'cancelled'
    progress: float = 0.0  # 0.0 to 1.0
    current_epoch: int = 0
    latest_loss: Optional[float] = None
    latest_accuracy: Optional[float] = None
    elapsed_time: float = 0.0  # seconds
    estimated_cost: float = 0.0
    message: str = ""


@dataclass
class JobResult:
    """Results from a completed job"""
    job_id: str
    success: bool
    final_accuracy: Optional[float] = None
    final_loss: Optional[float] = None
    total_epochs: int = 0
    training_time: float = 0.0  # seconds
    total_cost: float = 0.0
    logs: str = ""
    model_path: Optional[str] = None
    metrics_history: Dict[str, List[float]] = field(default_factory=dict)
    baseline_accuracy: Optional[float] = None
    passed_validation: bool = False
    error_message: str = ""


class GPUBackend(ABC):
    """Abstract base class for GPU backend implementations

    Implementations must handle:
    - Instance provisioning and management
    - Job upload and execution
    - Progress monitoring
    - Results retrieval
    - Cost tracking and limits
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize backend with configuration

        Args:
            config: Backend-specific configuration (API keys, limits, etc.)
        """
        self.config = config
        self.daily_budget = config.get('daily_budget', 5.0)
        self.max_job_cost = config.get('max_job_cost', 1.0)

    @abstractmethod
    def submit_job(self, job_config: JobConfig) -> str:
        """Submit a job for execution

        Args:
            job_config: Job configuration

        Returns:
            Job ID for tracking

        Raises:
            ValueError: If job exceeds budget limits
            ConnectionError: If backend is unreachable
        """
        pass

    @abstractmethod
    def get_status(self, job_id: str) -> JobStatus:
        """Get current status of a job

        Args:
            job_id: Job identifier

        Returns:
            Current job status
        """
        pass

    @abstractmethod
    def get_results(self, job_id: str) -> JobResult:
        """Get results from a completed job

        Args:
            job_id: Job identifier

        Returns:
            Job results including metrics and logs
        """
        pass

    @abstractmethod
    def estimate_cost(self, job_config: JobConfig) -> float:
        """Estimate cost for a job before running

        Args:
            job_config: Job configuration

        Returns:
            Estimated cost in USD
        """
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job

        Args:
            job_id: Job identifier

        Returns:
            True if successfully cancelled
        """
        pass

    @abstractmethod
    def list_available_gpus(self) -> List[Dict[str, Any]]:
        """List available GPU instances

        Returns:
            List of available GPU offers with specs and pricing
        """
        pass

    def validate_budget(self, estimated_cost: float) -> bool:
        """Check if job is within budget limits

        Args:
            estimated_cost: Estimated job cost

        Returns:
            True if within budget
        """
        return estimated_cost <= self.max_job_cost
