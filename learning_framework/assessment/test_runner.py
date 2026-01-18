"""Automated test runner for executing and analyzing pytest results."""

import subprocess
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Any


class TestRunner:
    """Run and analyze pytest test results."""

    def __init__(self):
        """Initialize the TestRunner."""
        pass

    def run_tests(self, test_file: str, timeout: int = 60) -> Dict[str, Any]:
        """
        Run pytest on a test file and capture results.

        Args:
            test_file: Path to the test file to run
            timeout: Maximum time in seconds to allow tests to run

        Returns:
            Dictionary containing:
                - passed (bool): Whether all tests passed
                - num_passed (int): Number of passed tests
                - num_failed (int): Number of failed tests
                - stdout (str): Captured stdout
                - stderr (str): Captured stderr
                - failures (list): List of failure descriptions
                - timeout (bool): Whether timeout occurred (optional)
        """
        # Check if file exists
        if not Path(test_file).exists():
            return {
                "passed": False,
                "num_passed": 0,
                "num_failed": 0,
                "stdout": "",
                "stderr": f"Test file not found: {test_file}",
                "failures": []
            }

        # Run pytest with appropriate flags
        cmd = [
            sys.executable, "-m", "pytest",
            test_file,
            "-v",  # Verbose output
            "-s",  # Don't capture output (show print statements)
            "--tb=short",  # Short traceback format
            "--color=no"  # Disable color codes for easier parsing
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path(test_file).parent.parent  # Run from project root
            )

            stdout = result.stdout
            stderr = result.stderr

            # Parse the output
            num_passed, num_failed = self._parse_pytest_output(stdout)
            failures = self._extract_failures(stdout)

            return {
                "passed": num_failed == 0 and num_passed > 0,
                "num_passed": num_passed,
                "num_failed": num_failed,
                "stdout": stdout,
                "stderr": stderr,
                "failures": failures
            }

        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "num_passed": 0,
                "num_failed": 0,
                "stdout": "Test execution timed out",
                "stderr": f"Tests exceeded timeout of {timeout} seconds",
                "failures": [],
                "timeout": True
            }

        except Exception as e:
            return {
                "passed": False,
                "num_passed": 0,
                "num_failed": 0,
                "stdout": "",
                "stderr": f"Error running tests: {str(e)}",
                "failures": []
            }

    def _parse_pytest_output(self, output: str) -> tuple[int, int]:
        """
        Parse pytest output to extract passed and failed test counts.

        Args:
            output: Pytest stdout output

        Returns:
            Tuple of (num_passed, num_failed)
        """
        num_passed = 0
        num_failed = 0

        # First try to parse from summary line like "2 passed, 1 failed in 0.05s"
        # This is the most reliable method
        summary_match = re.search(
            r'(\d+) failed.*?(\d+) passed',
            output
        )
        if summary_match:
            num_failed = int(summary_match.group(1))
            num_passed = int(summary_match.group(2))
            return num_passed, num_failed

        summary_match = re.search(
            r'(\d+) passed.*?(\d+) failed',
            output
        )
        if summary_match:
            num_passed = int(summary_match.group(1))
            num_failed = int(summary_match.group(2))
            return num_passed, num_failed

        # Only passed tests
        summary_match = re.search(r'(\d+) passed', output)
        if summary_match:
            num_passed = int(summary_match.group(1))
            num_failed = 0
            return num_passed, num_failed

        # Only failed tests
        summary_match = re.search(r'(\d+) failed', output)
        if summary_match:
            num_failed = int(summary_match.group(1))
            num_passed = 0
            return num_passed, num_failed

        # Fallback: Count PASSED and FAILED in test result lines only
        # Match patterns like "test_name PASSED" or "test_name FAILED"
        passed_matches = re.findall(r'::\w+\s+PASSED', output)
        num_passed = len(passed_matches)

        failed_matches = re.findall(r'::\w+\s+FAILED', output)
        num_failed = len(failed_matches)

        return num_passed, num_failed

    def _extract_failures(self, output: str) -> List[str]:
        """
        Extract failure information from pytest output.

        Args:
            output: Pytest stdout output

        Returns:
            List of failure descriptions (test names)
        """
        failures = []

        # Extract from FAILURES section if present
        # This is more reliable than counting FAILED keywords
        failures_section_match = re.search(
            r'=+ FAILURES =+(.+?)(?:=+ short test summary|=+ \d+ failed|$)',
            output,
            re.DOTALL
        )

        if failures_section_match:
            failures_text = failures_section_match.group(1)
            # Extract test names from failure section headers like "_______ test_name _______"
            test_failures = re.findall(r'_+ (test_\w+) _+', failures_text)
            failures.extend(test_failures)

        # If no failures section found, extract from test result lines
        if not failures:
            # Match patterns like "test_file.py::test_name FAILED"
            failed_matches = re.findall(r'::(test_\w+)\s+FAILED', output)
            failures.extend(failed_matches)

        return failures

    def run_with_coverage(self, test_file: str, timeout: int = 60) -> Dict[str, Any]:
        """
        Run pytest with coverage analysis.

        Args:
            test_file: Path to the test file to run
            timeout: Maximum time in seconds to allow tests to run

        Returns:
            Dictionary with test results and coverage information
        """
        # Check if file exists
        if not Path(test_file).exists():
            return {
                "passed": False,
                "num_passed": 0,
                "num_failed": 0,
                "stdout": "",
                "stderr": f"Test file not found: {test_file}",
                "failures": [],
                "coverage": None
            }

        # Run pytest with coverage
        cmd = [
            sys.executable, "-m", "pytest",
            test_file,
            "-v",
            "-s",
            "--tb=short",
            "--color=no",
            "--cov",  # Enable coverage
            "--cov-report=term"  # Terminal coverage report
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path(test_file).parent.parent
            )

            stdout = result.stdout
            stderr = result.stderr

            # Parse the output
            num_passed, num_failed = self._parse_pytest_output(stdout)
            failures = self._extract_failures(stdout)

            # Extract coverage percentage if available
            coverage = self._extract_coverage(stdout)

            return {
                "passed": num_failed == 0 and num_passed > 0,
                "num_passed": num_passed,
                "num_failed": num_failed,
                "stdout": stdout,
                "stderr": stderr,
                "failures": failures,
                "coverage": coverage
            }

        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "num_passed": 0,
                "num_failed": 0,
                "stdout": "Test execution timed out",
                "stderr": f"Tests exceeded timeout of {timeout} seconds",
                "failures": [],
                "timeout": True,
                "coverage": None
            }

        except Exception as e:
            # If coverage plugin not available, fall back to regular run
            return self.run_tests(test_file, timeout)

    def _extract_coverage(self, output: str) -> Optional[float]:
        """
        Extract coverage percentage from pytest-cov output.

        Args:
            output: Pytest stdout with coverage report

        Returns:
            Coverage percentage as float, or None if not found
        """
        # Look for coverage percentage like "TOTAL    95%"
        coverage_match = re.search(r'TOTAL\s+(\d+)%', output)
        if coverage_match:
            return float(coverage_match.group(1))

        return None
