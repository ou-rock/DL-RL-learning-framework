"""Results validation for GPU job outputs"""

from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

from .base import JobResult


@dataclass
class ValidationCheck:
    """Single validation check result"""
    name: str
    passed: bool
    message: str
    expected: Optional[Any] = None
    actual: Optional[Any] = None


class ResultsValidator:
    """Validates GPU job results against targets and baselines"""

    def __init__(self, tolerance: float = 0.05):
        """Initialize validator

        Args:
            tolerance: Acceptable difference from baseline (e.g., 0.05 = 5%)
        """
        self.tolerance = tolerance

    def validate_accuracy(
        self,
        result: JobResult,
        target: float
    ) -> Tuple[bool, str]:
        """Validate that accuracy meets target

        Args:
            result: Job result
            target: Target accuracy (0.0 to 1.0)

        Returns:
            (passed, message) tuple
        """
        if result.final_accuracy is None:
            return False, "No accuracy reported"

        if result.final_accuracy >= target:
            return True, f"Accuracy {result.final_accuracy:.2%} meets target {target:.2%}"
        else:
            diff = target - result.final_accuracy
            return False, f"Accuracy {result.final_accuracy:.2%} is {diff:.2%} below target {target:.2%}"

    def validate_against_baseline(
        self,
        result: JobResult
    ) -> Tuple[bool, str]:
        """Validate that accuracy is within tolerance of baseline

        Args:
            result: Job result with baseline_accuracy set

        Returns:
            (passed, message) tuple
        """
        if result.baseline_accuracy is None:
            return True, "No baseline to compare against"

        if result.final_accuracy is None:
            return False, "No accuracy reported"

        diff = abs(result.final_accuracy - result.baseline_accuracy)

        if diff <= self.tolerance:
            return True, (
                f"Accuracy {result.final_accuracy:.2%} within {self.tolerance:.0%} "
                f"of baseline {result.baseline_accuracy:.2%}"
            )
        else:
            return False, (
                f"Accuracy {result.final_accuracy:.2%} differs by {diff:.2%} "
                f"from baseline {result.baseline_accuracy:.2%} (tolerance: {self.tolerance:.0%})"
            )

    def validate_completion(self, result: JobResult) -> Tuple[bool, str]:
        """Validate that job completed successfully

        Args:
            result: Job result

        Returns:
            (passed, message) tuple
        """
        if not result.success:
            return False, f"Job failed: {result.error_message}"

        return True, "Job completed successfully"

    def validate_cost(
        self,
        result: JobResult,
        max_cost: float
    ) -> Tuple[bool, str]:
        """Validate that job stayed within cost budget

        Args:
            result: Job result
            max_cost: Maximum allowed cost

        Returns:
            (passed, message) tuple
        """
        if result.total_cost > max_cost:
            return False, f"Cost ${result.total_cost:.2f} exceeded budget ${max_cost:.2f}"

        return True, f"Cost ${result.total_cost:.2f} within budget ${max_cost:.2f}"

    def validate(
        self,
        result: JobResult,
        target_accuracy: float = 0.90,
        max_cost: Optional[float] = None
    ) -> Dict[str, Any]:
        """Run full validation and return report

        Args:
            result: Job result to validate
            target_accuracy: Target accuracy threshold
            max_cost: Maximum allowed cost (optional)

        Returns:
            Validation report with overall pass/fail and individual checks
        """
        checks: List[ValidationCheck] = []

        # Check completion
        passed, msg = self.validate_completion(result)
        checks.append(ValidationCheck(
            name="completion",
            passed=passed,
            message=msg
        ))

        # Check accuracy target
        passed, msg = self.validate_accuracy(result, target_accuracy)
        checks.append(ValidationCheck(
            name="accuracy_target",
            passed=passed,
            message=msg,
            expected=target_accuracy,
            actual=result.final_accuracy
        ))

        # Check baseline comparison
        passed, msg = self.validate_against_baseline(result)
        checks.append(ValidationCheck(
            name="baseline_comparison",
            passed=passed,
            message=msg,
            expected=result.baseline_accuracy,
            actual=result.final_accuracy
        ))

        # Check cost if limit provided
        if max_cost is not None:
            passed, msg = self.validate_cost(result, max_cost)
            checks.append(ValidationCheck(
                name="cost_limit",
                passed=passed,
                message=msg,
                expected=max_cost,
                actual=result.total_cost
            ))

        # Overall pass requires all checks to pass
        all_passed = all(check.passed for check in checks)

        return {
            'passed': all_passed,
            'job_id': result.job_id,
            'checks': [
                {
                    'name': c.name,
                    'passed': c.passed,
                    'message': c.message,
                    'expected': c.expected,
                    'actual': c.actual
                }
                for c in checks
            ],
            'summary': (
                "All validation checks passed" if all_passed
                else f"{sum(1 for c in checks if not c.passed)} check(s) failed"
            )
        }
