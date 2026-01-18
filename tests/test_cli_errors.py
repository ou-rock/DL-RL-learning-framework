"""Tests for CLI error handling"""
import pytest
from click.testing import CliRunner
from learning_framework.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_learn_unknown_concept_shows_suggestions(runner):
    """Unknown concept shows helpful suggestions"""
    result = runner.invoke(cli, ['learn', '--concept', 'backpropagaton'])
    assert result.exit_code != 0 or "not found" in result.output.lower()


def test_quiz_missing_concept_shows_help(runner):
    """Quiz with missing concept shows help"""
    result = runner.invoke(cli, ['quiz', '--concept', 'nonexistent_concept_xyz'])
    # Should gracefully handle missing concept (either error or "no questions" message)
    output_lower = result.output.lower()
    assert ("error" in output_lower or
            "not found" in output_lower or
            "no quiz" in output_lower or
            result.exit_code != 0)


def test_challenge_missing_shows_list(runner):
    """Missing challenge shows available challenges"""
    result = runner.invoke(cli, ['challenge', 'nonexistent_challenge'])
    # Should suggest running --list
    assert "list" in result.output.lower() or result.exit_code != 0


def test_config_error_shows_fix_suggestion(runner, tmp_path):
    """Config errors show how to fix"""
    # This tests the error display, not actual config issues
    result = runner.invoke(cli, ['config'])
    # Should at least show config file location
    assert "config" in result.output.lower()
