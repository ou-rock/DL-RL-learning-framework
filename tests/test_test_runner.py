"""Tests for TestRunner - automated test execution."""

import pytest
import os
import tempfile
from pathlib import Path
from learning_framework.assessment.test_runner import TestRunner


@pytest.fixture
def test_runner():
    """Create a TestRunner instance."""
    return TestRunner()


@pytest.fixture
def temp_test_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_test_runner_executes_pytest(test_runner, temp_test_dir):
    """Test that TestRunner can execute pytest on a simple passing test."""
    # Create a simple passing test file
    test_file = temp_test_dir / "test_simple.py"
    test_file.write_text("""
def test_simple_pass():
    assert 1 + 1 == 2

def test_another_pass():
    assert True
""")

    # Run the test
    result = test_runner.run_tests(str(test_file))

    # Verify results
    assert result["passed"] is True
    assert result["num_passed"] == 2
    assert result["num_failed"] == 0
    assert "test_simple_pass PASSED" in result["stdout"]
    assert "test_another_pass PASSED" in result["stdout"]


def test_test_runner_reports_failures(test_runner, temp_test_dir):
    """Test that TestRunner can detect and report failing tests."""
    # Create a test file with failures
    test_file = temp_test_dir / "test_failures.py"
    test_file.write_text("""
def test_will_pass():
    assert 1 + 1 == 2

def test_will_fail():
    assert 1 + 1 == 3, "Math is broken"

def test_another_fail():
    assert False, "This should fail"
""")

    # Run the test
    result = test_runner.run_tests(str(test_file))

    # Verify results
    assert result["passed"] is False
    assert result["num_passed"] == 1
    assert result["num_failed"] == 2
    assert len(result["failures"]) == 2
    assert "test_will_fail" in result["failures"][0]
    assert "test_another_fail" in result["failures"][1]


def test_test_runner_captures_output(test_runner, temp_test_dir):
    """Test that TestRunner captures stdout and stderr."""
    # Create a test file that produces output
    test_file = temp_test_dir / "test_output.py"
    test_file.write_text("""
import sys

def test_with_output():
    print("This is stdout output")
    sys.stderr.write("This is stderr output\\n")
    assert True
""")

    # Run the test
    result = test_runner.run_tests(str(test_file))

    # Verify output capture
    assert result["passed"] is True
    assert "stdout" in result
    assert "stderr" in result
    # Pytest captures print statements, so they appear in output
    assert "This is stdout output" in result["stdout"] or "This is stdout output" in result["stderr"]


def test_test_runner_handles_timeout(test_runner, temp_test_dir):
    """Test that TestRunner can timeout long-running tests."""
    # Create a test that takes too long
    test_file = temp_test_dir / "test_timeout.py"
    test_file.write_text("""
import time

def test_slow():
    time.sleep(10)
    assert True
""")

    # Run with short timeout
    result = test_runner.run_tests(str(test_file), timeout=1)

    # Should fail due to timeout
    assert result["passed"] is False
    assert "timeout" in result["stdout"].lower() or "timeout" in result["stderr"].lower() or result.get("timeout", False)


def test_test_runner_handles_nonexistent_file(test_runner):
    """Test that TestRunner handles non-existent test files gracefully."""
    result = test_runner.run_tests("/nonexistent/test_file.py")

    assert result["passed"] is False
    assert result["num_passed"] == 0
    assert result["num_failed"] == 0


def test_test_runner_with_coverage(test_runner, temp_test_dir):
    """Test that TestRunner can run tests with coverage analysis."""
    # Create a simple test file
    test_file = temp_test_dir / "test_coverage.py"
    test_file.write_text("""
def test_simple():
    assert 1 + 1 == 2
""")

    # Run with coverage (optional feature)
    result = test_runner.run_with_coverage(str(test_file))

    # Should still run successfully
    assert result["passed"] is True
    assert result["num_passed"] == 1
    # Coverage info might be present (or None if coverage not available)
    if "coverage" in result and result["coverage"] is not None:
        assert isinstance(result["coverage"], (int, float, str))


def test_test_runner_parses_pytest_output_format(test_runner, temp_test_dir):
    """Test that TestRunner correctly parses pytest's output format."""
    # Create a test with mixed results
    test_file = temp_test_dir / "test_mixed.py"
    test_file.write_text("""
def test_one():
    assert True

def test_two():
    assert False

def test_three():
    assert 1 == 1
""")

    # Run the test
    result = test_runner.run_tests(str(test_file))

    # Verify parsing
    assert result["num_passed"] == 2
    assert result["num_failed"] == 1
    assert result["passed"] is False
    assert isinstance(result["failures"], list)
    assert len(result["failures"]) == 1
