"""Centralized error handling with actionable suggestions"""

from typing import List, Optional, Dict, Any


class LearningFrameworkError(Exception):
    """Base exception for all learning framework errors"""

    def __init__(self, message: str, code: Optional[str] = None):
        self.message = message
        self.code = code or "E000"
        super().__init__(message)

    def __str__(self):
        return self.message


class ConfigurationError(LearningFrameworkError):
    """Error in configuration settings"""

    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message, code="E100")
        self.config_key = config_key


class ConceptNotFoundError(LearningFrameworkError):
    """Requested concept does not exist"""

    def __init__(self, concept: str, suggestions: Optional[List[str]] = None):
        message = f"Concept '{concept}' not found"
        if suggestions:
            message += f". Did you mean: {', '.join(suggestions)}?"
        super().__init__(message, code="E200")
        self.concept = concept
        self.suggestions = suggestions or []


class PrerequisiteError(LearningFrameworkError):
    """Missing prerequisites for a concept"""

    def __init__(self, concept: str, missing: List[str]):
        message = f"Cannot start '{concept}': missing prerequisites {missing}"
        super().__init__(message, code="E201")
        self.concept = concept
        self.missing = missing


class QuizError(LearningFrameworkError):
    """Error during quiz operation"""

    def __init__(self, message: str, concept: Optional[str] = None):
        super().__init__(message, code="E300")
        self.concept = concept


class ChallengeError(LearningFrameworkError):
    """Error during challenge operation"""

    def __init__(self, message: str, challenge_name: Optional[str] = None):
        super().__init__(message, code="E400")
        self.challenge_name = challenge_name


class GPUBackendError(LearningFrameworkError):
    """Error with GPU backend operation"""

    def __init__(self, message: str, backend: Optional[str] = None):
        super().__init__(message, code="E500")
        self.backend = backend


class BudgetExceededError(LearningFrameworkError):
    """Operation would exceed budget limits"""

    def __init__(
        self,
        requested: float,
        available: float,
        budget_type: str = "job"
    ):
        message = (
            f"Budget exceeded: requested ${requested:.2f}, "
            f"available ${available:.2f} ({budget_type} budget)"
        )
        super().__init__(message, code="E501")
        self.requested = requested
        self.available = available
        self.budget_type = budget_type


class VisualizationError(LearningFrameworkError):
    """Error rendering visualization"""

    def __init__(self, message: str, viz_type: Optional[str] = None):
        super().__init__(message, code="E600")
        self.viz_type = viz_type


class BuildError(LearningFrameworkError):
    """Error building C++ extensions"""

    def __init__(self, message: str, component: Optional[str] = None):
        super().__init__(message, code="E700")
        self.component = component


# Error formatting and suggestions

def format_error_message(error: LearningFrameworkError) -> str:
    """Format error for user-friendly display

    Args:
        error: The error to format

    Returns:
        Formatted error string with emoji and details
    """
    icon = "❌"

    if isinstance(error, ConfigurationError):
        icon = "⚙️"
    elif isinstance(error, ConceptNotFoundError):
        icon = "🔍"
    elif isinstance(error, PrerequisiteError):
        icon = "📚"
    elif isinstance(error, BudgetExceededError):
        icon = "💰"
    elif isinstance(error, GPUBackendError):
        icon = "🖥️"
    elif isinstance(error, BuildError):
        icon = "🔧"

    return f"{icon} Error [{error.code}]: {error.message}"


def suggest_fix(error: LearningFrameworkError) -> List[str]:
    """Provide actionable suggestions for fixing an error

    Args:
        error: The error to suggest fixes for

    Returns:
        List of suggestion strings
    """
    suggestions = []

    if isinstance(error, ConfigurationError):
        suggestions.append("Check your config file at user_data/config.yaml")
        if error.config_key:
            suggestions.append(f"Set the '{error.config_key}' key in your config")
        suggestions.append("Run 'lf config' to view current settings")

    elif isinstance(error, ConceptNotFoundError):
        suggestions.append("Run 'lf learn' to see available concepts")
        if error.suggestions:
            suggestions.append(f"Try: {error.suggestions[0]}")
        suggestions.append("Check spelling of concept name")

    elif isinstance(error, PrerequisiteError):
        suggestions.append(f"Complete these concepts first: {', '.join(error.missing)}")
        suggestions.append("Use 'lf learn --concept <name>' to start a prerequisite")

    elif isinstance(error, QuizError):
        suggestions.append("Ensure concept has quiz questions defined")
        suggestions.append("Run 'lf index' to refresh concept data")

    elif isinstance(error, ChallengeError):
        suggestions.append("Run 'lf challenge --list' to see available challenges")
        suggestions.append("Check that challenge file exists in data/challenges/")

    elif isinstance(error, BudgetExceededError):
        suggestions.append(f"Wait for daily budget reset (${error.available:.2f} available)")
        suggestions.append("Use '--estimate' flag to check cost before submitting")
        suggestions.append("Consider using a smaller dataset or fewer epochs")

    elif isinstance(error, GPUBackendError):
        suggestions.append("Check your API key in config")
        suggestions.append("Verify internet connection")
        suggestions.append("Try 'lf scale --list-gpus' to check availability")

    elif isinstance(error, BuildError):
        suggestions.append("Ensure CMake 3.15+ is installed")
        suggestions.append("Install pybind11: pip install pybind11")
        suggestions.append("Check C++ compiler is available (g++, clang++, or MSVC)")

    else:
        suggestions.append("Check the logs for more details")
        suggestions.append("Run with --verbose flag for more information")

    return suggestions


def handle_error(error: Exception) -> tuple[str, List[str]]:
    """Handle any error and return formatted message with suggestions

    Args:
        error: Any exception

    Returns:
        Tuple of (formatted_message, suggestions)
    """
    if isinstance(error, LearningFrameworkError):
        return format_error_message(error), suggest_fix(error)
    else:
        # Wrap unknown errors
        wrapped = LearningFrameworkError(str(error), code="E999")
        return format_error_message(wrapped), ["Check the error message for details"]
