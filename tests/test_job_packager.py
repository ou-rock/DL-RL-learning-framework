import pytest
import tempfile
import tarfile
from pathlib import Path
from learning_framework.backends.packager import JobPackager
from learning_framework.backends.base import JobConfig


@pytest.fixture
def sample_implementation(tmp_path):
    """Create a sample implementation file"""
    impl_file = tmp_path / "backprop.py"
    impl_file.write_text('''
def backward(x, y, weights):
    """Backpropagation implementation"""
    return {"W1": x, "W2": y}
''')
    return impl_file


@pytest.fixture
def job_config(sample_implementation):
    """Create a sample job config"""
    return JobConfig(
        concept="backpropagation",
        implementation_path=str(sample_implementation),
        dataset="cifar10",
        epochs=30,
        target_accuracy=0.90
    )


def test_packager_creates_tarball(job_config, tmp_path):
    """Packager creates a valid tarball"""
    packager = JobPackager(output_dir=tmp_path)
    bundle_path = packager.package(job_config)

    assert bundle_path.exists()
    assert bundle_path.suffix == '.gz'
    assert tarfile.is_tarfile(bundle_path)


def test_package_contains_implementation(job_config, tmp_path):
    """Package includes the implementation file"""
    packager = JobPackager(output_dir=tmp_path)
    bundle_path = packager.package(job_config)

    with tarfile.open(bundle_path, 'r:gz') as tar:
        names = tar.getnames()
        assert any('backprop.py' in name for name in names)


def test_package_contains_train_script(job_config, tmp_path):
    """Package includes generated training script"""
    packager = JobPackager(output_dir=tmp_path)
    bundle_path = packager.package(job_config)

    with tarfile.open(bundle_path, 'r:gz') as tar:
        names = tar.getnames()
        assert any('train.py' in name for name in names)


def test_package_contains_requirements(job_config, tmp_path):
    """Package includes requirements.txt"""
    packager = JobPackager(output_dir=tmp_path)
    bundle_path = packager.package(job_config)

    with tarfile.open(bundle_path, 'r:gz') as tar:
        names = tar.getnames()
        assert any('requirements.txt' in name for name in names)


def test_package_contains_config(job_config, tmp_path):
    """Package includes job configuration"""
    packager = JobPackager(output_dir=tmp_path)
    bundle_path = packager.package(job_config)

    with tarfile.open(bundle_path, 'r:gz') as tar:
        names = tar.getnames()
        assert any('job_config.json' in name for name in names)
