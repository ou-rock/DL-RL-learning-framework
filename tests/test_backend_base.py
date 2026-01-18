import pytest
from abc import ABC
from learning_framework.backends.base import GPUBackend, JobConfig, JobStatus


def test_gpu_backend_is_abstract():
    """GPUBackend cannot be instantiated directly"""
    with pytest.raises(TypeError):
        GPUBackend()


def test_job_config_creation():
    """JobConfig holds job configuration"""
    config = JobConfig(
        concept="backpropagation",
        implementation_path="/path/to/backprop.py",
        dataset="cifar10",
        epochs=30,
        batch_size=64,
        target_accuracy=0.90
    )
    assert config.concept == "backpropagation"
    assert config.epochs == 30
    assert config.target_accuracy == 0.90


def test_job_status_creation():
    """JobStatus holds job status information"""
    status = JobStatus(
        job_id="vast_12345",
        status="running",
        progress=0.45,
        current_epoch=15,
        latest_loss=0.234,
        latest_accuracy=0.85,
        elapsed_time=300.0,
        estimated_cost=0.05
    )
    assert status.status == "running"
    assert status.progress == 0.45


def test_backend_has_required_methods():
    """GPUBackend defines required abstract methods"""
    # Verify abstract methods exist
    assert hasattr(GPUBackend, 'submit_job')
    assert hasattr(GPUBackend, 'get_status')
    assert hasattr(GPUBackend, 'get_results')
    assert hasattr(GPUBackend, 'estimate_cost')
    assert hasattr(GPUBackend, 'cancel_job')
