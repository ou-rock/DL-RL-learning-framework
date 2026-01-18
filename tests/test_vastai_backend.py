import pytest
from unittest.mock import Mock, patch, MagicMock
from learning_framework.backends.vastai import VastaiBackend
from learning_framework.backends.base import JobConfig, JobStatus


@pytest.fixture
def mock_config():
    """Configuration for Vast.ai backend"""
    return {
        'api_key': 'test_api_key',
        'daily_budget': 5.0,
        'max_job_cost': 1.0,
        'preferred_gpu': 'RTX 3060',
        'min_gpu_ram': 8
    }


@pytest.fixture
def backend(mock_config):
    """Create Vast.ai backend with mocked API"""
    return VastaiBackend(mock_config)


@pytest.fixture
def job_config(tmp_path):
    """Sample job configuration"""
    impl = tmp_path / "backprop.py"
    impl.write_text("def backward(): pass")
    return JobConfig(
        concept="backpropagation",
        implementation_path=str(impl),
        dataset="cifar10",
        epochs=30,
        target_accuracy=0.90
    )


def test_backend_inherits_from_gpubackend(backend):
    """VastaiBackend inherits from GPUBackend"""
    from learning_framework.backends.base import GPUBackend
    assert isinstance(backend, GPUBackend)


def test_backend_has_api_key(backend):
    """Backend stores API key from config"""
    assert backend.api_key == 'test_api_key'


@patch('requests.get')
def test_list_available_gpus(mock_get, backend):
    """list_available_gpus returns GPU offers"""
    mock_response = Mock()
    mock_response.json.return_value = {
        'offers': [
            {'id': 1, 'gpu_name': 'RTX 3060', 'dph_total': 0.15},
            {'id': 2, 'gpu_name': 'RTX 3080', 'dph_total': 0.25}
        ]
    }
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    gpus = backend.list_available_gpus()
    assert len(gpus) >= 1
    assert 'gpu_name' in gpus[0]


def test_estimate_cost(backend, job_config):
    """estimate_cost returns estimated cost in USD"""
    cost = backend.estimate_cost(job_config)
    assert isinstance(cost, float)
    assert cost > 0


@patch('requests.get')
@patch('requests.put')
@patch('paramiko.SSHClient')
def test_submit_job_returns_job_id(mock_ssh, mock_put, mock_get, backend, job_config):
    """submit_job returns a job ID"""
    # Mock Vast.ai API responses
    mock_get.return_value.json.return_value = {
        'offers': [{'id': 123, 'gpu_name': 'RTX 3060', 'dph_total': 0.15}]
    }
    mock_get.return_value.status_code = 200

    mock_put.return_value.json.return_value = {
        'success': True,
        'new_contract': 456
    }
    mock_put.return_value.status_code = 200

    # Mock SSH connection
    mock_ssh_instance = MagicMock()
    mock_ssh.return_value = mock_ssh_instance
    mock_ssh_instance.exec_command.return_value = (None, MagicMock(), MagicMock())

    job_id = backend.submit_job(job_config)
    assert job_id.startswith('vast_')


def test_get_status_returns_job_status(backend):
    """get_status returns JobStatus object"""
    # Would need mock for real test
    # This tests the interface exists
    status = backend.get_status('vast_123')
    assert isinstance(status, JobStatus)
