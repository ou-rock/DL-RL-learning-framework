"""Tests for error handling framework"""
import pytest
from learning_framework.errors import (
    LearningFrameworkError,
    ConfigurationError,
    ConceptNotFoundError,
    PrerequisiteError,
    QuizError,
    ChallengeError,
    GPUBackendError,
    BudgetExceededError,
    format_error_message,
    suggest_fix
)


def test_base_error_has_message():
    """Base error includes message and code"""
    err = LearningFrameworkError("Something went wrong", code="E001")
    assert str(err) == "Something went wrong"
    assert err.code == "E001"


def test_configuration_error():
    """ConfigurationError for config issues"""
    err = ConfigurationError("Config file not found", config_key="api_key")
    assert "Config file" in str(err)
    assert err.config_key == "api_key"


def test_concept_not_found_error():
    """ConceptNotFoundError with suggestions"""
    err = ConceptNotFoundError(
        "backpropagaton",
        suggestions=["backpropagation", "propagation"]
    )
    assert "backpropagaton" in str(err)
    assert err.suggestions == ["backpropagation", "propagation"]


def test_prerequisite_error():
    """PrerequisiteError lists missing prerequisites"""
    err = PrerequisiteError(
        concept="neural_networks",
        missing=["linear_algebra", "calculus"]
    )
    assert err.concept == "neural_networks"
    assert "linear_algebra" in err.missing


def test_budget_exceeded_error():
    """BudgetExceededError includes cost info"""
    err = BudgetExceededError(
        requested=2.50,
        available=1.00,
        budget_type="daily"
    )
    assert err.requested == 2.50
    assert err.available == 1.00


def test_format_error_message_includes_emoji():
    """format_error_message creates user-friendly output"""
    err = ConceptNotFoundError("backprop")
    formatted = format_error_message(err)
    assert "❌" in formatted or "Error" in formatted


def test_suggest_fix_provides_actionable_steps():
    """suggest_fix provides actionable suggestions"""
    err = ConfigurationError("API key not set", config_key="vastai_api_key")
    suggestions = suggest_fix(err)
    assert len(suggestions) > 0
    assert any("config" in s.lower() or "set" in s.lower() for s in suggestions)
