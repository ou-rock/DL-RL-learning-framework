import pytest
from click.testing import CliRunner
from learning_framework.cli import cli


def test_cli_viz_help():
    """Test viz command shows help"""
    runner = CliRunner()
    result = runner.invoke(cli, ['viz', '--help'])
    assert result.exit_code == 0
    assert 'interactive visualization' in result.output.lower()


def test_cli_visualize_help():
    """Test visualize command shows help"""
    runner = CliRunner()
    result = runner.invoke(cli, ['visualize', '--help'])
    assert result.exit_code == 0
    assert 'visualization' in result.output.lower()
    assert 'concept' in result.output.lower()
