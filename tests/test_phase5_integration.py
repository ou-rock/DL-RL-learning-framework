"""Phase 5 integration tests for GPU backend system"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from learning_framework.backends import (
    GPUBackend,
    JobConfig,
    JobStatus,
    JobResult,
    JobPackager,
    CostController,
    VastaiBackend
)
from learning_framework.backends.validator import ResultsValidator


@pytest.fixture
def sample_implementation(tmp_path):
    """Create sample implementation for testing"""
    impl = tmp_path / "test_impl.py"
    impl.write_text('''
def forward(x, weights):
    return x @ weights

def backward(x, y, weights):
    return {"W": x.T @ y}
''')
    return impl


@pytest.fixture
def job_config(sample_implementation):
    """Create sample job configuration"""
    return JobConfig(
        concept="test_concept",
        implementation_path=str(sample_implementation),
        dataset="mnist",
        epochs=10,
        target_accuracy=0.95
    )


class TestFullWorkflow:
    """Test complete GPU job workflow"""

    def test_package_estimate_submit_validate_flow(self, job_config, tmp_path):
        """Test full workflow: package -> estimate -> submit -> validate"""

        # 1. Package job
        packager = JobPackager(output_dir=tmp_path)
        bundle = packager.package(job_config)
        assert bundle.exists()

        # 2. Cost controller checks budget
        controller = CostController(
            db_path=tmp_path / "costs.db",
            daily_budget=5.0,
            max_job_cost=1.0
        )
        assert controller.can_spend(0.50)

        # 3. Estimate cost (mocked)
        with patch.object(VastaiBackend, 'list_available_gpus') as mock_gpus:
            mock_gpus.return_value = [
                {'id': 1, 'gpu_name': 'RTX 3060', 'gpu_ram': 12, 'hourly_cost': 0.15}
            ]

            backend = VastaiBackend({
                'api_key': 'test_key',
                'daily_budget': 5.0,
                'max_job_cost': 1.0
            })

            estimated_cost = backend.estimate_cost(job_config)
            assert estimated_cost > 0

        # 4. Validate results
        result = JobResult(
            job_id="test_job",
            success=True,
            final_accuracy=0.96,
            baseline_accuracy=0.95,
            total_cost=0.25
        )

        validator = ResultsValidator(tolerance=0.05)
        report = validator.validate(result, target_accuracy=0.95)

        assert report['passed'] == True
        assert all(check['passed'] for check in report['checks'])

    def test_budget_enforcement(self, job_config, tmp_path):
        """Test that budget limits are enforced"""
        controller = CostController(
            db_path=tmp_path / "costs.db",
            daily_budget=1.0,
            max_job_cost=0.50
        )

        # Can spend within limits
        assert controller.can_spend(0.40)

        # Cannot exceed max job cost
        assert not controller.can_spend(0.60)

        # Record some spending
        controller.record_spending("job_1", 0.40)
        controller.record_spending("job_2", 0.40)

        # Cannot exceed daily budget
        assert not controller.can_spend(0.30)
        assert controller.can_spend(0.20)

    def test_validation_catches_failures(self):
        """Test that validator correctly identifies failures"""
        validator = ResultsValidator(tolerance=0.05)

        # Test accuracy below target
        result = JobResult(
            job_id="test",
            success=True,
            final_accuracy=0.80,
            baseline_accuracy=0.90
        )

        report = validator.validate(result, target_accuracy=0.90)
        assert report['passed'] == False

        # Find which checks failed
        failed = [c for c in report['checks'] if not c['passed']]
        assert len(failed) >= 1
        assert any('accuracy' in c['name'] for c in failed)


class TestBackendAbstraction:
    """Test that backend abstraction works correctly"""

    def test_all_backends_share_interface(self):
        """All backends implement required methods"""
        required_methods = [
            'submit_job',
            'get_status',
            'get_results',
            'estimate_cost',
            'cancel_job',
            'list_available_gpus'
        ]

        # VastaiBackend implements all required methods
        for method in required_methods:
            assert hasattr(VastaiBackend, method)

    def test_job_config_serialization(self, job_config):
        """JobConfig can be serialized for transmission"""
        from dataclasses import asdict
        import json

        config_dict = asdict(job_config)
        json_str = json.dumps(config_dict)

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed['concept'] == job_config.concept
        assert parsed['epochs'] == job_config.epochs


class TestPackagerIntegration:
    """Test job packager creates valid bundles"""

    def test_package_is_extractable(self, job_config, tmp_path):
        """Created package can be extracted and contains expected files"""
        import tarfile

        packager = JobPackager(output_dir=tmp_path)
        bundle = packager.package(job_config)

        # Extract and verify
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        with tarfile.open(bundle, 'r:gz') as tar:
            tar.extractall(extract_dir)

        project_dir = extract_dir / "project"
        assert project_dir.exists()
        assert (project_dir / "train.py").exists()
        assert (project_dir / "requirements.txt").exists()
        assert (project_dir / "job_config.json").exists()

    def test_train_script_is_valid_python(self, job_config, tmp_path):
        """Generated train script is valid Python"""
        import tarfile
        import ast

        packager = JobPackager(output_dir=tmp_path)
        bundle = packager.package(job_config)

        with tarfile.open(bundle, 'r:gz') as tar:
            train_file = tar.extractfile("project/train.py")
            train_content = train_file.read().decode('utf-8')

        # Verify it parses as valid Python
        ast.parse(train_content)


class TestCostControllerPersistence:
    """Test cost controller database persistence"""

    def test_spending_persists_across_instances(self, tmp_path):
        """Spending records persist when creating new controller instance"""
        db_path = tmp_path / "costs.db"

        # First instance records spending
        controller1 = CostController(db_path=db_path, daily_budget=5.0)
        controller1.record_spending("job_1", 1.50)

        # Second instance sees the spending
        controller2 = CostController(db_path=db_path, daily_budget=5.0)
        assert controller2.get_daily_spending() == 1.50
        assert controller2.get_remaining_budget() == 3.50
