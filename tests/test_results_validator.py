import pytest
from learning_framework.backends.validator import ResultsValidator
from learning_framework.backends.base import JobResult


@pytest.fixture
def validator():
    """Create results validator"""
    return ResultsValidator(tolerance=0.05)


def test_validate_accuracy_passes_when_meeting_target():
    """Validation passes when accuracy meets target"""
    validator = ResultsValidator(tolerance=0.05)

    result = JobResult(
        job_id="test_1",
        success=True,
        final_accuracy=0.92,
        baseline_accuracy=0.90
    )

    is_valid, message = validator.validate_accuracy(result, target=0.90)
    assert is_valid == True


def test_validate_accuracy_fails_when_below_target():
    """Validation fails when accuracy below target"""
    validator = ResultsValidator(tolerance=0.05)

    result = JobResult(
        job_id="test_1",
        success=True,
        final_accuracy=0.80,
        baseline_accuracy=0.90
    )

    is_valid, message = validator.validate_accuracy(result, target=0.90)
    assert is_valid == False
    assert "below target" in message.lower()


def test_validate_against_baseline_within_tolerance():
    """Validation passes when within tolerance of baseline"""
    validator = ResultsValidator(tolerance=0.05)

    result = JobResult(
        job_id="test_1",
        success=True,
        final_accuracy=0.88,
        baseline_accuracy=0.90
    )

    is_valid, message = validator.validate_against_baseline(result)
    assert is_valid == True


def test_validate_against_baseline_outside_tolerance():
    """Validation fails when too far from baseline"""
    validator = ResultsValidator(tolerance=0.05)

    result = JobResult(
        job_id="test_1",
        success=True,
        final_accuracy=0.70,
        baseline_accuracy=0.90
    )

    is_valid, message = validator.validate_against_baseline(result)
    assert is_valid == False


def test_full_validation_checks_all_criteria():
    """Full validation checks accuracy, baseline, and completion"""
    validator = ResultsValidator(tolerance=0.05)

    result = JobResult(
        job_id="test_1",
        success=True,
        final_accuracy=0.91,
        baseline_accuracy=0.90,
        total_epochs=30
    )

    report = validator.validate(result, target_accuracy=0.90)

    assert 'passed' in report
    assert 'checks' in report
    assert len(report['checks']) >= 2
