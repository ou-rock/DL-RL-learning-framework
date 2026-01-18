import pytest
from click.testing import CliRunner
from learning_framework.cli import cli


def test_learn_command_exists():
    """Test learn command is available"""
    runner = CliRunner()
    result = runner.invoke(cli, ['learn', '--help'])
    assert result.exit_code == 0
    assert 'learning' in result.output.lower()


def test_quiz_command_with_concept():
    """Test quiz command accepts concept option"""
    runner = CliRunner()
    result = runner.invoke(cli, ['quiz', '--help'])
    assert result.exit_code == 0
    assert '--concept' in result.output or 'concept' in result.output
