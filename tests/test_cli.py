import pytest
from click.testing import CliRunner
from learning_framework.cli import cli


def test_cli_help():
    """Test CLI displays help message"""
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'Interactive Deep Learning' in result.output


def test_cli_version():
    """Test CLI displays version"""
    runner = CliRunner()
    result = runner.invoke(cli, ['--version'])
    assert result.exit_code == 0
    assert '0.1.0' in result.output
